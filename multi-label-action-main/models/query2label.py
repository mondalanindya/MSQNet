import time
import timm
import math
import torch
from torch import nn, Tensor
from torch.optim import Adam
from utils.utils import AverageMeter
from torch.nn.parallel import DistributedDataParallel as DDP


class Query2Label(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        hidden_dim = 512
        self.backbone = timm.create_model('resnetv2_50', pretrained=True, num_classes=hidden_dim)
        self.pos_encod = PositionalEncoding(d_model=hidden_dim)
        self.transformer = nn.Transformer(d_model=hidden_dim, batch_first=True)
        # self.transformer.load_state_dict(torch.load('models/transformer.ckpt')['transformer']) # hidden_dim = 256
        self.query_embed = nn.Embedding(num_classes, hidden_dim)
        self.fc = GroupWiseLinear(num_classes, hidden_dim, bias=True)

    def forward(self, images):
        b, t, c, h, w = images.size()
        images = images.reshape(-1, c, h, w)
        x = self.backbone(images)
        x = x.reshape(b, t, -1)
        x = self.pos_encod(x)
        query_embed = self.query_embed.weight.unsqueeze(0).repeat(b, 1, 1)
        hs = self.transformer(x, query_embed) # b, t, d
        out = self.fc(hs)
        return out
    

class Query2LabelExecutor:
    def __init__(self, train_loader, test_loader, criterion, eval_metric, class_list, test_every, distributed, gpu_id) -> None:
        super().__init__()
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = criterion.to(gpu_id)
        self.eval_metric = eval_metric.to(gpu_id)
        self.class_list = class_list
        self.test_every = test_every
        self.distributed = distributed
        self.gpu_id = gpu_id
        q2l = Query2Label(num_classes=len(class_list)).to(gpu_id)
        if distributed: 
            self.q2l = DDP(q2l, device_ids=[gpu_id], find_unused_parameters=False)
        else: 
            self.q2l = q2l   
        for p in self.q2l.parameters():
            p.requires_grad = True
        self.optimizer = Adam([{"params": self.q2l.parameters(), "lr": 0.00001}])

    def _train_batch(self, data, label):
        self.optimizer.zero_grad()
        output = self.q2l(data)
        loss_this = self.criterion(output, label)
        loss_this.backward()
        self.optimizer.step()
        return loss_this.item()

    def _train_epoch(self, epoch):
        self.q2l.train()
        loss_meter = AverageMeter()
        start_time = time.time()
        for data, label in self.train_loader:
            data, label = data.to(self.gpu_id), label.to(self.gpu_id)
            loss_this = self._train_batch(data, label)
            loss_meter.update(loss_this, data.shape[0])
        elapsed_time = time.time() - start_time
        if (self.distributed and self.gpu_id == 0) or not self.distributed:
            print("Epoch [" + str(epoch + 1) + "]"
                  + "[" + str(time.strftime("%H:%M:%S", time.gmtime(elapsed_time))) + "]"
                  + " loss: " + "{:.4f}".format(loss_meter.avg))
    
    def train(self, start_epoch, end_epoch):
        for epoch in range(start_epoch, end_epoch):
            self._train_epoch(epoch)
            if (epoch + 1) % self.test_every == 0:
                eval = self.test()
                if (self.distributed and self.gpu_id == 0) or not self.distributed:
                    print("[INFO] Evaluation Metric: {:.2f}".format(eval * 100))
    
    def test(self):
        self.q2l.eval()
        eval_meter = AverageMeter()
        for data, label in self.test_loader:
            data, label = data.to(self.gpu_id), label.long().to(self.gpu_id)
            with torch.no_grad():
                output = self.q2l(data)
            eval_this = self.eval_metric(output, label)
            eval_meter.update(eval_this.item(), data.shape[0])
        return eval_meter.avg
    
    def save(self, file_path="./checkpoint.pth"):
        backbone_state_dict = self.q2l.backbone.state_dict()
        transformer_state_dict = self.q2l.transformer.state_dict()
        query_embed_state_dict = self.q2l.query_embed.state_dict()
        fc_state_dict = self.q2l.fc.state_dict()
        optimizer_state_dict = self.optimizer.state_dict()
        torch.save({"backbone": backbone_state_dict,
                    "transformer": transformer_state_dict,
                    "query_embed": query_embed_state_dict,
                    "fc": fc_state_dict,
                    "optimizer": optimizer_state_dict},
                    file_path)
        
    def load(self, file_path):
        checkpoint = torch.load(file_path)
        self.q2l.backbone.load_state_dict(checkpoint["backbone"])
        self.q2l.transformer.load_state_dict(checkpoint["transformer"])
        self.q2l.query_embed.load_state_dict(checkpoint["query_embed"])
        self.q2l.fc.load_state_dict(checkpoint["fc"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
    

class PositionalEncoding(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 50):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:, :x.size(1)]
        out = self.dropout(x)
        return out


class GroupWiseLinear(nn.Module):
    # could be changed to: 
    # output = torch.einsum('ijk,zjk->ij', x, self.W)
    # or output = torch.einsum('ijk,jk->ij', x, self.W[0])
    def __init__(self, num_class, hidden_dim, bias=True):
        super().__init__()
        self.num_class = num_class
        self.hidden_dim = hidden_dim
        self.bias = bias

        self.W = nn.Parameter(torch.Tensor(1, num_class, hidden_dim))
        if bias:
            self.b = nn.Parameter(torch.Tensor(1, num_class))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(2))
        for i in range(self.num_class):
            self.W[0][i].data.uniform_(-stdv, stdv)
        if self.bias:
            for i in range(self.num_class):
                self.b[0][i].data.uniform_(-stdv, stdv)

    def forward(self, x):
        # x: B,K,d
        x = (self.W * x).sum(-1)
        if self.bias:
            x = x + self.b
        return x