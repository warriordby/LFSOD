import torch
import torch.distributed as dist
import torch.nn as nn
import math
import warnings
from torch.nn.modules.utils import _pair as to_2tuple

from mmcv.cnn import build_norm_layer
from mmcv.runner import BaseModule
from mmcv.cnn.bricks import DropPath
from mmcv.cnn.utils.weight_init import (constant_init, normal_init,
                                        trunc_normal_init)
from mmcv.utils import Registry
from mmcv.cnn import MODELS as MMCV_MODELS
MODELS = Registry('models', parent=MMCV_MODELS)

BACKBONES = MODELS

class Mlp(BaseModule):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features#256 512 1024 2048
        #全连接层线性变换到指定维度hidden_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        #定义深度可分离卷积层 dwconv，对隐藏特征进行特征提取，通常用于减少参数数量并提高计算效率。
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)

        x = self.dwconv(x)
        x = self.act(x)#激活函数
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x

# 模型的初始卷积层,降采样和特征提取
class StemConv(BaseModule):
    def __init__(self, in_channels, out_channels, norm_cfg=dict(type='SyncBN', requires_grad=True)):
        super(StemConv, self).__init__()
        #创建一个顺序容器
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2,
                      kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            build_norm_layer(norm_cfg, out_channels // 2)[1],
            nn.GELU(), # GELU 激活函数，用于引入非线性变换
            nn.Conv2d(out_channels // 2, out_channels,
                      kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),#padding进行填充保证尺寸符合
            build_norm_layer(norm_cfg, out_channels)[1],
        )

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.size()
        x = x.flatten(2).transpose(1, 2)
        return x, H, W


class AttentionModule(BaseModule):
    def __init__(self, dim):
        super().__init__()
        #定义了一个 5x5 的深度可分离卷积层 conv0。指对输入特征的每个通道进行卷积操作，然后对通道之间的特征进行组合
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        #定义了两个的深度可分离卷积层，分别用于在水平和垂直方向上对输入特征进行局部关注
        self.conv0_1 = nn.Conv2d(dim, dim, (1, 7), padding=(0, 3), groups=dim)
        self.conv0_2 = nn.Conv2d(dim, dim, (7, 1), padding=(3, 0), groups=dim)

        self.conv1_1 = nn.Conv2d(dim, dim, (1, 11), padding=(0, 5), groups=dim)
        self.conv1_2 = nn.Conv2d(dim, dim, (11, 1), padding=(5, 0), groups=dim)

        self.conv2_1 = nn.Conv2d(
            dim, dim, (1, 21), padding=(0, 10), groups=dim)
        self.conv2_2 = nn.Conv2d(
            dim, dim, (21, 1), padding=(10, 0), groups=dim)
        #定义了一个 1x1 的卷积层 conv3，用于整合不同范围的局部关注结果，并将其与输入特征相结合
        self.conv3 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)

        attn_0 = self.conv0_1(attn)
        attn_0 = self.conv0_2(attn_0)

        attn_1 = self.conv1_1(attn)
        attn_1 = self.conv1_2(attn_1)

        attn_2 = self.conv2_1(attn)
        attn_2 = self.conv2_2(attn_2)
        attn = attn + attn_0 + attn_1 + attn_2

        attn = self.conv3(attn)

        return attn * u


class SpatialAttention(BaseModule):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()#GELU 激活函数，用于增加模型的非线性特性。
        self.spatial_gating_unit = AttentionModule(d_model)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)

    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut
        return x


class Block(BaseModule):

    def __init__(self,
                 dim,
                 mlp_ratio=4.,
                 drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_cfg=dict(type='SyncBN', requires_grad=True)):
        super().__init__()
        self.norm1 = build_norm_layer(norm_cfg, dim)[1]
        self.attn = SpatialAttention(dim)
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()#0000
        self.norm2 = build_norm_layer(norm_cfg, dim)[1]
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)
        layer_scale_init_value = 1e-2
        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)#定义了一个可学习的缩放因子参数
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.permute(0, 2, 1).view(B, C, H, W)
        x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1)
                               * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1)
                               * self.mlp(self.norm2(x)))
        x = x.view(B, C, N).permute(0, 2, 1)
        return x


class OverlapPatchEmbed(BaseModule):
    """ Image to Patch Embedding
    """

    def __init__(self, patch_size=7, stride=4, in_chans=3, embed_dim=768, norm_cfg=dict(type='SyncBN', requires_grad=True)):
        super().__init__()
        patch_size = to_2tuple(patch_size)

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = build_norm_layer(norm_cfg, embed_dim)[1]

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = self.norm(x)

        x = x.flatten(2).transpose(1, 2)

        return x, H, W


@ BACKBONES.register_module()
class MSCAN(BaseModule):
    def __init__(self,
                 in_chans=3,
                 embed_dims=[64, 128, 256, 512],
                 mlp_ratios=[4, 4, 4, 4],
                 drop_rate=0.,
                 drop_path_rate=0.,
                 depths=[3, 4, 6, 3],
                 num_stages=4,
                 norm_cfg=dict(type='SyncBN', requires_grad=True),
                 pretrained=None,
                 init_cfg=None):
        super(MSCAN, self).__init__(init_cfg=init_cfg)

        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be set at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is not None:
            raise TypeError('pretrained must be a str or None')
            
                # 指定分布式通信的后端和s初始化分布式进程组的方法
        backend = 'gloo'  # 或 'nccl'
        init_method = 'tcp://localhost:12345'  # 或其他适当的初始化方法

        # 初始化分布式进程组
        dist.init_process_group(backend=backend, init_method=init_method,rank=0,world_size=1) # world_size的值可能有影响

        self.depths = depths
        self.num_stages = num_stages

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate,
                                                sum(depths))]  # stochastic depth decay rule
        cur = 0

        for i in range(num_stages):
            if i == 0:
                patch_embed = StemConv(3, embed_dims[0], norm_cfg=norm_cfg)
            else:
                patch_embed = OverlapPatchEmbed(patch_size=7 if i == 0 else 3,
                                                stride=4 if i == 0 else 2,
                                                in_chans=in_chans if i == 0 else embed_dims[i - 1],
                                                embed_dim=embed_dims[i],
                                                norm_cfg=norm_cfg)

            block = nn.ModuleList([Block(dim=embed_dims[i], mlp_ratio=mlp_ratios[i],
                                         drop=drop_rate, drop_path=dpr[cur + j],
                                         norm_cfg=norm_cfg)
                                   for j in range(depths[i])])
            norm = nn.LayerNorm(embed_dims[i])
            cur += depths[i]
            conv_last = nn.Conv2d(embed_dims[i], 1, kernel_size=3, padding=1)

            setattr(self, f'conv_last{i+1}', conv_last)
            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)

    def init_weights(self):
        print('init cfg', self.init_cfg)
        if self.init_cfg is None:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_init(m, std=.02, bias=0.)
                elif isinstance(m, nn.LayerNorm):
                    constant_init(m, val=1.0, bias=0.)
                elif isinstance(m, nn.Conv2d):
                    fan_out = m.kernel_size[0] * m.kernel_size[
                        1] * m.out_channels
                    fan_out //= m.groups
                    normal_init(
                        m, mean=0, std=math.sqrt(2.0 / fan_out), bias=0)
        else:

            super(MSCAN, self).init_weights()

    def forward(self, x):
        if isinstance(x,list):
            x=x[0]
        B = x.shape[0]
        outs = []
        outs1=[]

        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            x, H, W = patch_embed(x)
            for blk in block:
                x = blk(x, H, W)
            x = norm(x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            outs.append(x)

            conv_last=getattr(self, f'conv_last{i+1}')
            outs1.append(conv_last(x))

        return outs#, outs1[0], outs1[1], outs1[2], outs1[3]


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        x = self.dwconv(x)
        return x
    
class mascan_b(MSCAN):
    def __init__(self, in_chans=3,
                  embed_dims=[64, 128, 320, 512], 
                  mlp_ratios=[4, 4, 4, 4], drop_rate=0, 
                  drop_path_rate=0, depths=[3, 4, 6, 3], 
                  num_stages=4, 
                  norm_cfg=dict(type='SyncBN', requires_grad=True),
                  pretrained=None, 
                  init_cfg=None):
        super().__init__(in_chans, embed_dims, mlp_ratios, drop_rate, drop_path_rate, depths, num_stages, norm_cfg, pretrained, init_cfg)

if __name__=='__main__':
    model=MSCAN()
    x=torch.randn(36, 3, 256, 256)

    model = model.to('cuda')
    x = x.to('cuda')


    y=model(x)