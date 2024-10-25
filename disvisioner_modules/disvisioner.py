import math
import warnings
from torch import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. " "The distribution of values may be incorrect.", stacklevel=2)

    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        tensor.clamp_(min=a, max=b)
        return tensor

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)

def drop_path(x: Tensor, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0], ) + (1, ) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self.apply(self._init_weights)
        nn.init.constant_(self.proj.weight.data, 0.0)
        if qkv_bias:
            nn.init.constant_(self.proj.bias.data, 0.0)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, q: Tensor, k: Tensor, v: Tensor):
        B_q, N_q, C_q = q.size()
        B_k, N_k, C_k = k.size()
        q = self.q(q).reshape(B_q, N_q, self.num_heads, -1).permute(0, 2, 1, 3)
        k = self.k(k).reshape(B_k, N_k, self.num_heads, -1).permute(0, 2, 1, 3)
        v = self.v(v).reshape(B_k, N_k, self.num_heads, -1).permute(0, 2, 1, 3)
        attn = self.attn_drop(F.softmax((q @ k.transpose(-2, -1)) * self.scale, dim=-1))
        q = (attn @ v).transpose(1, 2).reshape(q.size(0), q.size(2), -1)
        q = self.proj_drop(self.proj(q))
        return q

class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x: Tensor):
        return drop_path(x, self.drop_prob, self.training)

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x: Tensor):
        return x

class GELU(nn.Module):
    def __init__(self):
        super(GELU, self).__init__()

    def forward(self, input: torch.Tensor):
        return F.gelu(input)

class Mlp(nn.Module):
    def __init__(self, in_features: int, hidden_features=None, out_features=None, act_layer=GELU, drop=0.):
        super(Mlp, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=True)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=True)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Decoder(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=False, drop=0., attn_drop=0., drop_path=0.):
        super().__init__()
        self.self_attn = Attention(dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.cross_attn = Attention(dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else Identity()
        self.bn1 = nn.LayerNorm(dim)
        self.bn2 = nn.LayerNorm(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=2 * dim, out_features=dim, drop=drop)

    def forward(self, q, x):
        q_bn = self.bn1(q)
        q = q + self.drop_path(self.cross_attn(q_bn, x, x))
        q = q + self.drop_path(self.mlp(q))
        q_bn = self.bn2(q)
        q = q + self.drop_path(self.self_attn(q_bn, q_bn, q_bn))
        return q

class DisVisioner(nn.Module):
    def __init__(self,
                 image_hidden_size=1024,
                 text_hidden_size=768,
                 output_dim=768,
                 token_num=4,
                 num_refine=2,
                 ) -> None:
        super().__init__()
        self.text_class_mapping = nn.Sequential(nn.Linear(text_hidden_size, text_hidden_size),
                                            nn.LayerNorm(text_hidden_size),
                                            nn.LeakyReLU(),
                                            nn.Linear(text_hidden_size, output_dim),)
        
        self.image_local_mapping = nn.Sequential(nn.Linear(image_hidden_size, image_hidden_size),
                                            nn.LayerNorm(image_hidden_size),
                                            nn.LeakyReLU(),
                                            nn.Linear(image_hidden_size, image_hidden_size),
                                            nn.LayerNorm(image_hidden_size),
                                            nn.LeakyReLU(),
                                            nn.Linear(image_hidden_size, output_dim))
        
        self.q = nn.Parameter(torch.randn(1,token_num,output_dim))
        self.token_norm = nn.Sequential(nn.Linear(output_dim, output_dim), nn.LayerNorm(output_dim))
        self.decoder = nn.ModuleList([Decoder(output_dim, num_heads=8, qkv_bias=True) for _ in range(num_refine)])
        self.main_token_index = 0

    # def set_q_attn_save_dir(self,attn_save):
    #     self.q_attn_save = attn_save
    
    # def set_token_attn_save_dir(self,attn_token_save):
    #     self.token_attn_save = attn_token_save

    def forward(self, x, class_feat, return_attns=False):
        bs = x.shape[0]

        class_feat = torch.unsqueeze(class_feat,1)
        class_feat = self.text_class_mapping(class_feat)

        q = self.q.repeat(bs,1,1)
        q = torch.cat((q,class_feat),dim=1)

        local_output = x[:,1:,:]
        local_feat = self.image_local_mapping(local_output)

        attns = F.softmax(torch.bmm(q, local_feat.permute(0, 2, 1)), dim=1) 

        # q_attn_save = getattr(self, 'q_attn_save', None)
        # if q_attn_save is not None:
        #     os.makedirs(self.q_attn_save, exist_ok=True)
        #     torch.save(attns, os.path.join(self.q_attn_save,'q-attn.pt'))

        token = self.token_norm(torch.bmm(attns, local_feat))

        for dec in self.decoder:
            token = dec(token, local_feat)
            
        if return_attns:
            return token, attns[:,0], attns[:,1]
        else:
            return token
