# --------------------------------------------------------
# References:
# VideoMAE: https://github.com/MCG-NJU/VideoMAE
# timm: https://github.com/rwightman/pytorch-image-models
# --------------------------------------------------------
import os
import math
import torch
import numpy as np
import torch.nn as nn
from functools import partial
import torch.nn.functional as F
from collections import OrderedDict
from timm.models.registry import register_model
from timm.models.layers import drop_path, to_2tuple, trunc_normal_


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 400, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': (0.5, 0.5, 0.5), 'std': (0.5, 0.5, 0.5),
        **kwargs
    }


class Adapter(nn.Module):
    def __init__(self,                 
                 d_model=768,
                 bottleneck=64,
                 dropout=0.0,
                 init_option="bert",
                 adapter_scalar="1.0",
                 adapter_layernorm_option="in"):
        super().__init__()
        self.n_embd = d_model
        self.down_size = bottleneck

        #_before
        self.adapter_layernorm_option = adapter_layernorm_option

        self.adapter_layer_norm_before = None
        if adapter_layernorm_option == "in" or adapter_layernorm_option == "out":
            self.adapter_layer_norm_before = nn.LayerNorm(self.n_embd)

        if adapter_scalar == "learnable_scalar":
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.scale = float(adapter_scalar)

        self.down_proj = nn.Linear(self.n_embd, self.down_size)
        self.non_linear_func = nn.ReLU()
        self.up_proj = nn.Linear(self.down_size, self.n_embd)

        self.dropout = dropout
        if init_option == "bert":
            raise NotImplementedError
        elif init_option == "lora":
            with torch.no_grad():
                nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
                nn.init.zeros_(self.up_proj.weight)
                nn.init.zeros_(self.down_proj.bias)
                nn.init.zeros_(self.up_proj.bias)

    def forward(self, x, add_residual=True, residual=None):
        residual = x if residual is None else residual
        if self.adapter_layernorm_option == 'in':
            x = self.adapter_layer_norm_before(x)

        down = self.down_proj(x)
        down = self.non_linear_func(down)
        down = nn.functional.dropout(down, p=self.dropout, training=self.training)
        up = self.up_proj(down)

        up = up * self.scale

        if self.adapter_layernorm_option == 'out':
            up = self.adapter_layer_norm_before(up)

        if add_residual:
            output = up + residual
        else:
            output = up

        return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
    
    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
                 proj_drop=0., attn_head_dim=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.head_dim = head_dim
        self.scale = qk_scale or head_dim ** -0.5
        # unbind q, k, v linear project
        self.q_proj = nn.Linear(dim, all_head_dim, bias=False)
        self.v_proj = nn.Linear(dim, all_head_dim, bias=False)
        self.k_proj = nn.Linear(dim, all_head_dim, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(self, x):
        B, N, C = x.shape

        q = F.linear(input=x, weight=self.q_proj.weight, bias=self.q_bias)  # q = self.q_proj(x)
        _k = F.linear(input=x, weight=self.k_proj.weight, bias=None)  # k = self.k_proj(x
        k = self._shape(_k, N, B).view(B * self.num_heads, -1, self.head_dim)
        _v = F.linear(input=x, weight=self.v_proj.weight, bias=self.v_bias)
        v = self._shape(_v, N, B).view(B * self.num_heads, -1, self.head_dim)
        q = self._shape(q, N, B).view(B * self.num_heads, -1, self.head_dim)

        ################################
        q = q * self.scale  # fix: q scaling before prefix concat
        attn_weights = torch.bmm(q, k.transpose(1, 2))

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_probs = self.attn_drop(attn_weights)
        attn_output = torch.bmm(attn_probs, v)

        attn_output = attn_output.view(B, self.num_heads, N, self.head_dim)
        attn_output = attn_output.transpose(1, 2).reshape(B, N, -1)

        x = self.proj(attn_output)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, attn_head_dim=None):
        super().__init__()
        self.ffn_adapt = True
        self.ffn_option = 'parallel'
        self.ffn_num = 64
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        # rewrite FFN here
        self.fc1 = nn.Linear(dim, mlp_hidden_dim)
        self.fc2 = nn.Linear(mlp_hidden_dim, dim)
        self.act = act_layer()
        self.mlp_drop = nn.Dropout(drop)
        if self.ffn_adapt:
            self.adaptmlp = Adapter(d_model=768, dropout=drop, bottleneck=self.ffn_num,
                                    init_option="lora", adapter_scalar="0.1", adapter_layernorm_option="in")

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        if self.ffn_adapt and self.ffn_option == 'parallel':
            adapt_x = self.adaptmlp(x, add_residual=False)

        residual = x
        x = self.act(self.fc1(self.norm2(x)))
        x = self.drop_path(self.mlp_drop(self.fc2(x)))

        if self.ffn_adapt:
            if self.ffn_option == 'sequential':
                x = self.adaptmlp(x)
            elif self.ffn_option == 'parallel':
                x = x + adapt_x
            else:
                raise ValueError(self.ffn_adapt)
        x = residual + x
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, num_frames=16, tubelet_size=2):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.tubelet_size = int(tubelet_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0]) * (num_frames // self.tubelet_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv3d(in_channels=in_chans, out_channels=embed_dim, 
                              kernel_size = (self.tubelet_size, patch_size[0], patch_size[1]), 
                              stride=(self.tubelet_size, patch_size[0], patch_size[1]))

    def forward(self, x, **kwargs):
        B, C, T, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x
    
# sin-cos position encoding
# https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/Models.py#L31
def get_sinusoid_encoding_table(n_position, d_hid): 
    ''' Sinusoid position encoding table ''' 
    # TODO: make it with torch instead of numpy 
    def get_position_angle_vec(position): 
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)] 

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)]) 
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2]) # dim 2i 
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2]) # dim 2i+1 

    return torch.FloatTensor(sinusoid_table).unsqueeze(0) 


class AdaptFormer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, 
                 img_size=224, 
                 patch_size=16, 
                 in_chans=3, 
                 embed_dim=768, 
                 depth=12,
                 num_heads=12, 
                 mlp_ratio=4., 
                 qkv_bias=True, 
                 qk_scale=None, 
                 drop_rate=0., 
                 attn_drop_rate=0.,
                 drop_path_rate=0., 
                 norm_layer=nn.LayerNorm, 
                 use_learnable_pos_emb=False, 
                 all_frames=16,
                 tubelet_size=2,
                 vpt_on=False,
                 vpt_num=1,
                 name_ckpt='pretrain_vit_base_1600.pth'):
        super().__init__()
        self.depth = depth
        self.num_heads = num_heads
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.tubelet_size = tubelet_size
        self.vpt_on = vpt_on
        self.vpt_num = vpt_num
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, num_frames=all_frames, tubelet_size=self.tubelet_size)
        num_patches = self.patch_embed.num_patches

        if use_learnable_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        else:
            # sine-cosine positional embeddings is on the way
            self.pos_embed = get_sinusoid_encoding_table(num_patches, embed_dim)

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer) for i in range(depth)
                  ])
        
        self.norm =  norm_layer(embed_dim)

        if use_learnable_pos_emb:
            trunc_normal_(self.pos_embed, std=.02)

        self.apply(self._init_weights)

        if self.vpt_on:
            assert self.vpt_num > 0, self.vpt_num
            # properly registered
            self.embeddings = nn.ParameterList(  # batch, num_prompt, embed_dim
                [nn.Parameter(torch.empty(1, self.vpt_num, embed_dim)) for _ in range(depth)])
            for eee in self.embeddings:
                torch.nn.init.xavier_uniform_(eee.data)

        self.load_ckpt(name_ckpt)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def forward_features(self, x):
        x = self.patch_embed(x)
        B, _, _ = x.size()
        if self.pos_embed is not None:
            x = x + self.pos_embed.expand(B, -1, -1).type_as(x).to(x.device).clone().detach()
        x = self.pos_drop(x)
        for idx, blk in enumerate(self.blocks):
            if self.vpt_on:
                eee = self.embeddings[idx].expand(B, -1, -1)
                x = torch.cat([eee, x], dim=1)
            x = blk(x)
            if self.vpt_on:
                x = x[:, self.vpt_num:, :]
        return x

    def forward(self, x):
        # x = B x T x C x H x W
        x = x.transpose(1, 2)
        # x = B x C x T x H x W
        x = self.forward_features(x)
        return x
    
    def load_ckpt(self, name_ckpt):
        cur_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        path_ckpt = os.path.join(cur_path, 'checkpoint', name_ckpt)
        if not os.path.exists(path_ckpt):
            import wget
            os.makedirs(os.path.join(os.path.join(cur_path, 'checkpoint')), exist_ok=True)
            if name_ckpt == 'pretrain_vit_base_1600.pth':
                path_url = 'https://github.com/ShoufaChen/AdaptFormer/releases/download/v0.1/videomae_pretrain_vit_b_1600.pth'
            wget.download(path_url, path_ckpt)

        checkpoint = torch.load(path_ckpt, map_location='cpu')
        
        print("Load pre-trained checkpoint from: %s" % path_ckpt)
        if 'model' in checkpoint:
            raw_checkpoint_model = checkpoint['model']
        elif 'module' in checkpoint:
            raw_checkpoint_model = checkpoint['module']
        else:
            raw_checkpoint_model = checkpoint

        # TODO: refine
        if os.path.basename(path_ckpt).startswith('pretrain'):
            checkpoint_model = OrderedDict()
            for k, v in raw_checkpoint_model.items():
                if k.startswith('encoder.'):
                    checkpoint_model[k[8:]] = v  # remove 'encoder.' prefix
            del checkpoint_model['norm.weight']
            del checkpoint_model['norm.bias']
        elif os.path.basename(path_ckpt).startswith('finetune'):
            checkpoint_model = raw_checkpoint_model
        elif os.path.basename(path_ckpt) == "vit_base_patch16_224_in21k_tongzhan_new.pth":
            checkpoint_model = raw_checkpoint_model
            del checkpoint_model['norm.weight']
            del checkpoint_model['norm.bias']
        elif os.path.basename(path_ckpt) == "vit_base_patch16_224_in21k_to_video_tz.pth":
            checkpoint_model = raw_checkpoint_model
            # del checkpoint_model['norm.weight']
            # del checkpoint_model['norm.bias']
        elif os.path.basename(path_ckpt).startswith('swin_base_patch244'):
            checkpoint_model = OrderedDict()
            for k, v in raw_checkpoint_model['state_dict'].items():
                if k.startswith('backbone.'):
                    checkpoint_model[k[9:]] = v
        else:
            raise ValueError("Warning: Double Check!")

        state_dict = self.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # interpolate position embedding
        self.interpolate_pos_embed(checkpoint_model)

        # load pre-trained model
        self.msg = self.load_state_dict(checkpoint_model, strict=False)
        
    def interpolate_pos_embed(self, checkpoint_model):
        if 'pos_embed' in checkpoint_model:
            pos_embed_checkpoint = checkpoint_model['pos_embed']
            embedding_size = pos_embed_checkpoint.shape[-1]
            num_patches = self.patch_embed.num_patches
            num_extra_tokens = self.pos_embed.shape[-2] - num_patches
            # height (== width) for the checkpoint position embedding
            orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
            # height (== width) for the new position embedding
            new_size = int(num_patches ** 0.5)
            # class_token and dist_token are kept unchanged
            if orig_size != new_size:
                print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
                extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
                # only the position tokens are interpolated
                pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
                pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
                pos_tokens = torch.nn.functional.interpolate(
                    pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
                pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
                new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
                checkpoint_model['pos_embed'] = new_pos_embed


if __name__ == '__main__':
    adapt_former = AdaptFormer(all_frames=8, num_classes=157)
    # load(adapt_former, "models/pretrain_vit_b_1600.pth")
    breakpoint()
    x = torch.rand(12, 8, 3, 224, 224)
    y = adapt_former(x)
    print(x.shape)
    print(y.shape)