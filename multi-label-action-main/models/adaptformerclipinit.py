import time
import math
import torch
from torch import nn, Tensor
from torch.optim import Adam
import torch.nn.functional as F
from utils.utils import AverageMeter
from .adaptformer import AdaptFormer
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from transformers import CLIPTokenizer, CLIPTextModel, CLIPVisionModel, logging


class AdaptFormerCLIPInit(nn.Module):
    def __init__(self, class_embed, num_frames):
        super().__init__()
        self.num_frames = num_frames
        self.num_classes, self.embed_dim = class_embed.shape
        self.backbone = AdaptFormer(all_frames=num_frames, name_ckpt='pretrain_vit_base_1600.pth')
        self.norm = nn.LayerNorm(self.backbone.num_features)
        self.linear1 = nn.Linear(in_features=self.backbone.embed_dim, out_features=self.embed_dim, bias=False)
        self.pos_encod = PositionalEncoding(d_model=self.embed_dim)
        self.transformer = nn.Transformer(d_model=self.embed_dim, batch_first=True)
        self.query_embed = nn.Parameter(class_embed)
        self.group_linear = GroupWiseLinear(self.num_classes, self.embed_dim, bias=True)
        # self.image_model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
        # self.linear2 = nn.Linear(in_features=self.image_model.config.hidden_size + self.embed_dim, out_features=self.embed_dim, bias=False)

    def forward(self, images):
        b, t, c, h, w = images.size()
        x = self.backbone(images)
        x = self.norm(F.adaptive_avg_pool1d(x.transpose(1, 2), t).transpose(1, 2))
        x = self.linear1(x)
        x = self.pos_encod(x)
        # video_features = self.image_model(images.reshape(b*t, c, h, w))[1].reshape(b, t, -1).mean(dim=1, keepdim=True)
        # query_embed = self.linear2(torch.concat((self.query_embed.unsqueeze(0).repeat(b, 1, 1), video_features.repeat(1, self.num_classes, 1)), 2))
        query_embed = self.query_embed.unsqueeze(0).repeat(b, 1, 1)
        hs = self.transformer(x, query_embed) # b, t, d
        out = self.group_linear(hs)
        return out
    

class AdaptFormerCLIPInitExecutor:
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
        num_frames = self.train_loader.dataset[0][0].shape[0]
        logging.set_verbosity_error()
        class_embed = self._get_text_features(class_list)
        model = AdaptFormerCLIPInit(class_embed, num_frames).to(gpu_id)
        if distributed: 
            self.model = DDP(model, device_ids=[gpu_id])
        else: 
            self.model = model
        # train everything other than self.model.backbone and self.model.image_model
        for name, p in self.model.named_parameters():
            if not name.startswith('backbone') or not name.startswith('image_model'):
                p.requires_grad = True
            else:
                p.requires_grad = False
        # within the backbone only train the missing keys in self.model.backbone
        for name, p in self.model.backbone.named_parameters():
            if name in self.model.backbone.msg.missing_keys:
                p.requires_grad = True
            else:
                p.requires_grad = False
        self.optimizer = self._construct_optimizer()
        self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=10, eta_min=1e-8)

    def _construct_optimizer(self):
        optim_params = []
        for p in self.model.parameters():
            if p.requires_grad:
                optim_params.append(p)
        return Adam(optim_params, lr=0.01, weight_decay=0.)

    @staticmethod
    def _get_prompt(cl_names):
        temp_prompt = []
        for c in cl_names:
            temp_prompt.append(c)
        return temp_prompt
    
    def _get_text_features(self, cl_names):
        text_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        act_prompt = self._get_prompt(cl_names)
        texts = tokenizer(act_prompt, padding=True, return_tensors="pt")
        text_class = text_model(**texts).pooler_output.detach()
        return text_class

    def _train_batch(self, data, label):
        self.optimizer.zero_grad()
        output = self.model(data)
        loss_this = self.criterion(output, label)
        loss_this.backward()
        self.optimizer.step()
        return loss_this.item()

    def _train_epoch(self, epoch):
        self.model.train()
        loss_meter = AverageMeter()
        start_time = time.time()
        for data, label in self.train_loader:
            data, label = data.to(self.gpu_id, non_blocking=True), label.to(self.gpu_id, non_blocking=True)
            loss_this = self._train_batch(data, label)
            loss_meter.update(loss_this, data.shape[0])
        elapsed_time = time.time() - start_time
        if (self.distributed and self.gpu_id == 0) or not self.distributed:
            print("Epoch [" + str(epoch + 1) + "]"
                  + "[" + str(time.strftime("%H:%M:%S", time.gmtime(elapsed_time))) + "]"
                  + " loss: " + "{:.4f}".format(loss_meter.avg), flush=True)
    
    def train(self, start_epoch, end_epoch):
        for epoch in range(start_epoch, end_epoch):
            self._train_epoch(epoch)
            self.scheduler.step()
            if (epoch + 1) % self.test_every == 0:
                eval = self.test()
                if (self.distributed and self.gpu_id == 0) or not self.distributed:
                    print("[INFO] Evaluation Metric: {:.2f}".format(eval * 100), flush=True)
    
    def test(self):
        self.model.eval()
        eval_meter = AverageMeter()
        for data, label in self.test_loader:
            data, label = data.to(self.gpu_id), label.long().to(self.gpu_id)
            with torch.no_grad():
                output = self.model(data)
            eval_this = self.eval_metric(output, label)
            eval_meter.update(eval_this.item(), data.shape[0])
        return eval_meter.avg
    
    def save(self, file_path="./checkpoint.pth"):
        backbone_state_dict = self.model.backbone.state_dict()
        linear_state_dict = self.model.linear.state_dict()
        transformer_state_dict = self.model.transformer.state_dict()
        query_embed_state_dict = self.model.query_embed.state_dict()
        group_linear_state_dict = self.model.fc.state_dict()
        optimizer_state_dict = self.optimizer.state_dict()
        torch.save({"backbone": backbone_state_dict,
                    "linear": linear_state_dict,
                    "transformer": transformer_state_dict,
                    "query_embed": query_embed_state_dict,
                    "group_linear": group_linear_state_dict,
                    "optimizer": optimizer_state_dict},
                    file_path)

    def load(self, file_path):
        checkpoint = torch.load(file_path)
        self.model.backbone.load_state_dict(checkpoint["backbone"])
        self.model.linear.load_state_dict(checkpoint["linear"])
        self.model.transformer.load_state_dict(checkpoint["transformer"])
        self.model.query_embed.load_state_dict(checkpoint["query_embed"])
        self.model.group_linear.load_state_dict(checkpoint["group_linear"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])


class PositionalEncoding(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 2000):
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