import time
import math
import torch
import numpy as np
from torch import nn, Tensor
from torch.optim import Adam
import torch.nn.functional as F
from utils.utils import AverageMeter
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from transformers import VideoMAEModel, CLIPTokenizer, CLIPTextModel, logging


class VideoMAECLIPInit(nn.Module):
    def __init__(self, class_embed, num_frames):
        super().__init__()
        self.num_classes, self.embed_dim = class_embed.shape
        self.backbone = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics", num_frames=num_frames, ignore_mismatched_sizes=True)
        self.layer_norm = nn.LayerNorm(self.backbone.config.hidden_size)
        self.linear = nn.Linear(in_features=self.backbone.config.hidden_size, out_features=self.embed_dim, bias=False)
        self.pos_encod = PositionalEncoding(d_model=self.embed_dim)
        self.query_embed = nn.Parameter(class_embed)
        self.transformer = nn.Transformer(d_model=self.embed_dim, batch_first=True)
        self.group_linear = GroupWiseLinear(self.num_classes, self.embed_dim, bias=True)

    def forward(self, images):
        b, t, c, h, w = images.size()
        x = self.backbone(images)[0]
        x = self.layer_norm(F.adaptive_avg_pool1d(x.transpose(1, 2), t).transpose(1, 2))
        x = self.linear(x)
        x = self.pos_encod(x)
        query_embed = self.query_embed.unsqueeze(0).repeat(b, 1, 1)
        hs = self.transformer(x, query_embed) # b, t, d
        out = self.group_linear(hs)
        return out
    

class VideoMAECLIPInitExecutor:
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
        model = VideoMAECLIPInit(class_embed, num_frames).to(gpu_id)
        if distributed: 
            self.model = DDP(model, device_ids=[gpu_id])
        else: 
            self.model = model
        for p in self.model.parameters():
            p.requires_grad = True
        params = []
        for p in self.model.parameters():
            if p.requires_grad:
                params.append(p)
        self.optimizer = Adam([{"params": params, "lr": 0.00001}])
        self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=10)

    @staticmethod
    def _get_prompt(cl_names):
        temp_prompt = []
        for c in cl_names:
            temp_prompt.append(c)
        return temp_prompt
    
    def _get_text_features(self, cl_names):
        text_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch16")
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch16")
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
        self.scheduler.step()
        if (self.distributed and self.gpu_id == 0) or not self.distributed:
            print("Epoch [" + str(epoch + 1) + "]"
                  + "[" + str(time.strftime("%H:%M:%S", time.gmtime(elapsed_time))) + "]"
                  + " loss: " + "{:.4f}".format(loss_meter.avg), flush=True)
    
    def train(self, start_epoch, end_epoch):
        for epoch in range(start_epoch, end_epoch):
            self._train_epoch(epoch)
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


# sin-cos position encoding
# https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/Models.py#L31
def get_sinusoid_encoding_table(n_position, d_hid):
    """Sinusoid position encoding table"""

    # TODO: make it with torch instead of numpy
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)