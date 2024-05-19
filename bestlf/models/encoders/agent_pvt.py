# --------------------------------------------------------
# Agent Attention: On the Integration of Softmax and Linear Attention
# Modified by Tianzhu Ye
# -----------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

# from agent_utils.logger import get_root_logger
from mmcv.runner import _load_checkpoint

# from mmcv.utils import Registry
# from mmcv.cnn import MODELS as MMCV_MODELS
# MODELS = Registry('models', parent=MMCV_MODELS)

# BACKBONES = MODELS
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class AgentAttention(nn.Module):
    def __init__(self, dim, num_patches, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 sr_ratio=1, kernel_size=3, agent_num=49, downstream_agent_shape=(7, 7), scale=-0.5, **kwargs):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_patches = num_patches
        window_size = (int(num_patches ** 0.5), int(num_patches ** 0.5))
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** scale

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)


        self.kernel_size = kernel_size
        self.agent_num = agent_num

        self.dwc = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=kernel_size,
                                padding=kernel_size // 2, groups=dim)
        self.an_bias = nn.Parameter(torch.zeros(num_heads, agent_num, 7, 7))
        self.na_bias = nn.Parameter(torch.zeros(num_heads, agent_num, 7, 7))

        self.ah_bias = nn.Parameter(torch.zeros(1, num_heads, agent_num, window_size[0] // sr_ratio, 1))
        self.aw_bias = nn.Parameter(torch.zeros(1, num_heads, agent_num, 1, window_size[1] // sr_ratio))
        self.ha_bias = nn.Parameter(torch.zeros(1, num_heads, window_size[0], 1, agent_num))
        self.wa_bias = nn.Parameter(torch.zeros(1, num_heads, 1, window_size[1], agent_num))
        trunc_normal_(self.an_bias, std=.02)
        trunc_normal_(self.na_bias, std=.02)
        trunc_normal_(self.ah_bias, std=.02)
        trunc_normal_(self.aw_bias, std=.02)
        trunc_normal_(self.ha_bias, std=.02)
        trunc_normal_(self.wa_bias, std=.02)
        pool_size = int(agent_num ** 0.5)
        self.pool_size = pool_size
        self.downstream_agent_shape = downstream_agent_shape
        self.pool = nn.AdaptiveAvgPool2d(output_size=downstream_agent_shape)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, H, W):
        b, n, c = x.shape
        num_heads = self.num_heads
        head_dim = c // num_heads
        q = self.q(x)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(b, c, H, W)
            x_ = self.sr(x_).reshape(b, c, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(b, -1, 2, c).permute(2, 0, 1, 3)
        else:
            kv = self.kv(x).reshape(b, -1, 2, c).permute(2, 0, 1, 3)
        k, v = kv[0], kv[1]

        # following code is under the condition of self.sr_ratio == 1
        assert self.sr_ratio == 1
        downstream_agent_num = self.downstream_agent_shape[0] * self.downstream_agent_shape[1]
        agent_tokens = self.pool(q.reshape(b, H, W, c).permute(0, 3, 1, 2)).reshape(b, c, -1).permute(0, 2, 1)
        q = q.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        k = k.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        v = v.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        agent_tokens = agent_tokens.reshape(b, downstream_agent_num, num_heads, head_dim).permute(0, 2, 1, 3)

        # interpolate hw
        position_bias1 = nn.functional.interpolate(self.an_bias, size=(H, W), mode='bilinear')
        position_bias1 = position_bias1.reshape(num_heads, self.pool_size, self.pool_size, H * W).permute(0, 3, 1, 2)
        # interpolate agent_num
        position_bias1 = nn.functional.interpolate(position_bias1, size=self.downstream_agent_shape, mode='bilinear')
        position_bias1 = position_bias1.reshape(num_heads, H * W, downstream_agent_num).permute(0, 2, 1)
        position_bias1 = position_bias1.reshape(1, num_heads, downstream_agent_num, H * W).repeat(b, 1, 1, 1)
        
        # interpolate hw
        position_bias2 = nn.functional.interpolate((self.ah_bias + self.aw_bias).squeeze(0), size=(H, W), mode='bilinear')
        position_bias2 = position_bias2.reshape(num_heads, self.pool_size, self.pool_size, H * W).permute(0, 3, 1, 2)
        # interpolate agent_num
        position_bias2 = nn.functional.interpolate(position_bias2, size=self.downstream_agent_shape, mode='bilinear')
        position_bias2 = position_bias2.reshape(num_heads, H * W, downstream_agent_num).permute(0, 2, 1)
        position_bias2 = position_bias2.reshape(1, num_heads, downstream_agent_num, H * W).repeat(b, 1, 1, 1)
        
        position_bias = position_bias1 + position_bias2
        agent_attn = self.softmax((agent_tokens * self.scale) @ k.transpose(-2, -1) + position_bias)
        agent_attn = self.attn_drop(agent_attn)
        agent_v = agent_attn @ v

        # interpolate hw
        agent_bias1 = nn.functional.interpolate(self.na_bias, size=(H, W), mode='bilinear')
        agent_bias1 = agent_bias1.reshape(num_heads, self.pool_size, self.pool_size, H * W).permute(0, 3, 1, 2)
        # interpolate agent_num
        agent_bias1 = nn.functional.interpolate(agent_bias1, size=self.downstream_agent_shape, mode='bilinear')
        agent_bias1 = agent_bias1.reshape(1, num_heads, H * W, downstream_agent_num).repeat(b, 1, 1, 1)
        
        # interpolate hw
        agent_bias2 = (self.ha_bias + self.wa_bias).squeeze(0).permute(0, 3, 1, 2)
        agent_bias2 = nn.functional.interpolate(agent_bias2, size=(H, W), mode='bilinear')
        agent_bias2 = agent_bias2.reshape(num_heads, self.pool_size, self.pool_size, H * W).permute(0, 3, 1, 2)
        # interpolate agent_num
        agent_bias2 = nn.functional.interpolate(agent_bias2, size=self.downstream_agent_shape, mode='bilinear')
        agent_bias2 = agent_bias2.reshape(1, num_heads, H * W, downstream_agent_num).repeat(b, 1, 1, 1)
        agent_bias = agent_bias1 + agent_bias2
        q_attn = self.softmax((q * self.scale) @ agent_tokens.transpose(-2, -1) + agent_bias)
        q_attn = self.attn_drop(q_attn)
        x = q_attn @ agent_v

        x = x.transpose(1, 2).reshape(b, n, c)
        v = v.transpose(1, 2).reshape(b, H, W, c).permute(0, 3, 1, 2)
        x = x + self.dwc(v).permute(0, 2, 3, 1).reshape(b, n, c)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_patches, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1,
                 agent_num=49, downstream_agent_shape=(7, 7), kernel_size=3, attn_type='B', scale=-0.5):
        super().__init__()
        self.norm1 = norm_layer(dim)
        assert attn_type in ['A', 'B']# 条件为假时自爆
        if attn_type == 'A':
            self.attn = AgentAttention(
                dim, num_patches,
                num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio,
                agent_num=agent_num, downstream_agent_shape=downstream_agent_shape, kernel_size=kernel_size, scale=scale)
        else:
            self.attn = Attention(
                dim,
                num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        # assert img_size[0] % patch_size[0] == 0 and img_size[1] % patch_size[1] == 0, \
        #     f"img_size {img_size} should be divided by patch_size {patch_size}."
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W #3136 784 196 49
        #对于输入图像 x经过卷积操作后，它的每个位置都对应一个补丁。卷积核的滑动步幅是 patch_size，因此每次卷积核滑动的距离就是一个补丁的大小。这就实现了将图像分割成补丁的效果
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)# 
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape

        x = self.proj(x).flatten(2).transpose(1, 2) #卷积后分割成补丁C 
        x = self.norm(x)
        H, W = H // self.patch_size[0], W // self.patch_size[1] # 4 2 2 2

        return x, (H, W)

# @BACKBONES.register_module()
class AgentPVT(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], agent_sr_ratios='1111', num_stages=4,
                 agent_num=[9, 16, 49, 49], downstream_agent_shapes=[(3, 3), (4, 4), (7, 7), (7, 7)], 
                 kernel_size=3, attn_type='AAAA', scale=-0.5, init_cfg=None, **kwargs):
        super().__init__()
        self.init_cfg = init_cfg
        self.num_classes = num_classes
        self.depths = depths
        self.num_stages = num_stages

        self.agent_num = agent_num
        self.downstream_agent_shapes = downstream_agent_shapes
        self.attn_type = attn_type




        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0

        attn_type = 'AAAA' if attn_type is None else attn_type
        for i in range(num_stages):
            patch_embed = PatchEmbed(img_size=img_size if i == 0 else img_size // (2 ** (i - 1) * patch_size),# 
                                     patch_size=patch_size if i == 0 else 2, # 4 2 2 2
                                    #  patch_size=7 if i==0 else 3,
                                     in_chans=in_chans if i == 0 else embed_dims[i - 1], #3 64 128 256
                                     embed_dim=embed_dims[i]) # 64 128 256 512
            num_patches = patch_embed.num_patches
            print(img_size)
            #仅初始化
            pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dims[i]))  #位置嵌入，为模型提供输入数据的位置信息
            pos_drop = nn.Dropout(p=drop_rate)

            block = nn.ModuleList([Block(
                dim=embed_dims[i], num_patches=num_patches, num_heads=num_heads[i], mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias,
                qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + j],
                norm_layer=norm_layer, sr_ratio=sr_ratios[i] if attn_type[i] == 'B' else int(agent_sr_ratios[i]),
                agent_num=int(agent_num[i]),  downstream_agent_shape = downstream_agent_shapes[i],
                kernel_size=kernel_size, attn_type=attn_type[i], scale=scale)
                for j in range(depths[i])])
            cur += depths[i]

            conv_last = nn.Conv2d(embed_dims[i], 1, kernel_size=3, padding=1)
            setattr(self, f"conv_last{i+1}", conv_last)

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"pos_embed{i + 1}", pos_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"pos_drop{i + 1}", pos_drop)


        # init weights
        for i in range(num_stages):
            pos_embed = getattr(self, f"pos_embed{i + 1}")
            trunc_normal_(pos_embed, std=.02)
        # trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)
        # self.init_weights(pretrained)

    def init_weights(self, pretrained=None):
        # logger = get_root_logger()
        # assert 'checkpoint' in self.init_cfg, f'Only support ' \
        #                                         f'specify `Pretrained` in ' \
        #                                         f'`init_cfg` in ' \
        #                                         f'{self.__class__.__name__} '
        # ckpt = _load_checkpoint(
        #     self.init_cfg.checkpoint, logger=logger, map_location='cpu')
        # if 'state_dict' in ckpt:
        #     state_dict = ckpt['state_dict']
        # elif 'model' in ckpt:
        #     state_dict = ckpt['model']
        # else:
        #     state_dict = ckpt
        # load state_dict
        # missing_keys, unexpected_keys = self.load_state_dict(state_dict, False)
        # print('missing keys:', missing_keys)
        # print('unexpected keys:', unexpected_keys)
        if isinstance(pretrained, str):
            print('from hello{} load pretrained...'.format(pretrained))
            self.load_state_dict(torch.load(pretrained),strict=False)



    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

#######存在疑问
    def _get_pos_embed(self, pos_embed, patch_embed, H, W):
        # if H * W == self.patch_embed1.num_patches:# 3136
        if H * W == patch_embed.num_patches:#pos_embed.num_patches==patch_embed.num_patches
            return pos_embed
        else:
            return F.interpolate(
                pos_embed.reshape(1, patch_embed.H, patch_embed.W, -1).permute(0, 3, 1, 2),
                size=(H, W), mode="bilinear").reshape(1, -1, H * W).permute(0, 2, 1)

    def forward_features(self, x):
        outs = []
        outs1=[]

        B = x.shape[0]

        for i in range(self.num_stages):# 4 对四个阶段补丁层进行处理
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            pos_embed = getattr(self, f"pos_embed{i + 1}")
            pos_drop = getattr(self, f"pos_drop{i + 1}")
            block = getattr(self, f"block{i + 1}")
            x, (H, W) = patch_embed(x) #分割补丁层  输出为B H*W C
            #检查pos维度 
            pos_embed = self._get_pos_embed(pos_embed, patch_embed, H, W)

            x = pos_drop(x + pos_embed)#随机置零
            for blk in block:
                x = blk(x, H, W)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            conv_last=getattr(self, f"conv_last{i+1}")


            outs.append(x)
            out=conv_last(x)
            outs1.append(out)          

        return outs, tuple(outs1)

    def forward(self, x):
        if isinstance(x, list):
            x_rgb=x[0]
        else:
            x_rgb=x
        outs ,outs1 = self.forward_features(x_rgb)

        return outs ,outs1[0] ,outs1[1] ,outs1[2] ,outs1[3]

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(AgentPVT, self).train(mode)

def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v

    return out_dict

class agent_pvt_v2_b2(AgentPVT):
    def __init__(self, **kwargs):
        super(agent_pvt_v2_b2, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], 
        pretrain=r'/root/autodl-tmp/LFSOD/bestlf/Pretain_weight/fpn_agent_pvt_s_12-16-28-28.pth'  ,**kwargs)

# class agent_pvt_v2_b2(AgentPVT):
#     def __init__(self, **kwargs):
#         super(agent_pvt_v2_b2, self).__init__(
#             patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4],
#             qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1],
#             drop_rate=0.0, drop_path_rate=0.1)

# def agent_pvt_tiny(pretrained=False, **kwargs):
#     model = PyramidVisionTransformer(
#         patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
#         **kwargs)
#     model.default_cfg = _cfg()

#     return model


# def agent_pvt_tiny_2x(pretrained=False, **kwargs):
#     model = PyramidVisionTransformer(
#         patch_size=2, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
#         **kwargs)
#     model.default_cfg = _cfg()

#     return model


# def agent_pvt_small(pretrained=False, **kwargs):
#     model = PyramidVisionTransformer(
#         patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], **kwargs)
#     model.default_cfg = _cfg()

#     return model


# def agent_pvt_medium(pretrained=False, **kwargs):
#     model = PyramidVisionTransformer(
#         patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 18, 3], sr_ratios=[8, 4, 2, 1],
#         **kwargs)
#     model.default_cfg = _cfg()

#     return model


# def agent_pvt_large(pretrained=False, **kwargs):
#     model = PyramidVisionTransformer(
#         patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 8, 27, 3], sr_ratios=[8, 4, 2, 1],
#         **kwargs)
#     model.default_cfg = _cfg()

#     return model
if __name__=='__main__':
    x=torch.rand(36,3,224,224)
    model= agent_pvt_v2_b2()
    y=model(x)
