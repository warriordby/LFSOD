# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu

# Agent Attention: On the Integration of Softmax and Linear Attention
# Modified by Tianzhu Ye
# -----------------------------------------------------------------------
from functools import partial
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.nn.functional as F
import math
from agent_builder import BACKBONES
from agent_utils.logger import get_root_logger
from mmcv.runner import _load_checkpoint

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


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class AgentAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, input_resolution, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., shift_size=0,
                 kernel_size=3, agent_num=49, downstream_agent_shape=(7, 7), scale=-0.5, **kwargs):

        super().__init__()
        self.dim = dim
        # input_resolution is the resolution for cls task
        self.input_resolution = input_resolution
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** scale
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)
        self.shift_size = shift_size

        self.kernel_size = kernel_size
        self.agent_num = agent_num

        self.dwc = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=kernel_size,
                                padding=kernel_size // 2, groups=dim)
        self.an_bias = nn.Parameter(torch.zeros(num_heads, agent_num, 7, 7))
        self.na_bias = nn.Parameter(torch.zeros(num_heads, agent_num, 7, 7))

        self.ah_bias = nn.Parameter(torch.zeros(1, num_heads, agent_num, input_resolution[0], 1))
        self.aw_bias = nn.Parameter(torch.zeros(1, num_heads, agent_num, 1, input_resolution[1]))
        self.ha_bias = nn.Parameter(torch.zeros(1, num_heads, input_resolution[0], 1, agent_num))
        self.wa_bias = nn.Parameter(torch.zeros(1, num_heads, 1, input_resolution[1], agent_num))
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

    def forward(self, x, mask=None, hw_shape=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """

        b, n, c = x.shape
        h, w = hw_shape
        assert h * w == n
        num_heads = self.num_heads
        head_dim = c // num_heads
        qkv = self.qkv(x).reshape(b, n, 3, c).permute(2, 0, 1, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        # q, k, v: b, n, c

        downstream_agent_num = self.downstream_agent_shape[0] * self.downstream_agent_shape[1]
        #池化，代理令牌A
        agent_tokens = self.pool(q.reshape(b, h, w, c).permute(0, 3, 1, 2)).reshape(b, c, -1).permute(0, 2, 1)
        q = q.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        k = k.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        v = v.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        agent_tokens = agent_tokens.reshape(b, downstream_agent_num, num_heads, head_dim).permute(0, 2, 1, 3)

        # interpolate hw
        position_bias1 = nn.functional.interpolate(self.an_bias, size=hw_shape, mode='bilinear')
        position_bias1 = position_bias1.reshape(num_heads, self.pool_size, self.pool_size, h * w).permute(0, 3, 1, 2)
        
        # interpolate agent_num
        position_bias1 = nn.functional.interpolate(position_bias1, size=self.downstream_agent_shape, mode='bilinear')
        position_bias1 = position_bias1.reshape(num_heads, h * w, downstream_agent_num).permute(0, 2, 1)
        position_bias1 = position_bias1.reshape(1, num_heads, downstream_agent_num, h * w).repeat(b, 1, 1, 1)
        
        # interpolate hw
        position_bias2 = (self.ah_bias + self.aw_bias).squeeze(0)
        position_bias2 = nn.functional.interpolate(position_bias2, size=hw_shape, mode='bilinear')
        # position_bias2 = nn.functional.interpolate((self.ah_bias + self.aw_bias).squeeze(0), size=hw_shape, mode='bilinear')
        position_bias2 = position_bias2.reshape(num_heads, self.pool_size, self.pool_size, h * w).permute(0, 3, 1, 2)
        
        # interpolate agent_num
        position_bias2 = nn.functional.interpolate(position_bias2, size=self.downstream_agent_shape, mode='bilinear')
        position_bias2 = position_bias2.reshape(num_heads, h * w, downstream_agent_num).permute(0, 2, 1)
        position_bias2 = position_bias2.reshape(1, num_heads, downstream_agent_num, h * w).repeat(b, 1, 1, 1)
        
        position_bias = position_bias1 + position_bias2
        agent_attn = self.softmax((agent_tokens * self.scale) @ k.transpose(-2, -1) + position_bias)
        agent_attn = self.attn_drop(agent_attn)
        agent_v = agent_attn @ v #代理特征V_A

        # interpolate hw
        agent_bias1 = nn.functional.interpolate(self.na_bias, size=hw_shape, mode='bilinear')
        agent_bias1 = agent_bias1.reshape(num_heads, self.pool_size, self.pool_size, h * w).permute(0, 3, 1, 2)
        
        # interpolate agent_num
        agent_bias1 = nn.functional.interpolate(agent_bias1, size=self.downstream_agent_shape, mode='bilinear')
        agent_bias1 = agent_bias1.reshape(1, num_heads, h * w, downstream_agent_num).repeat(b, 1, 1, 1)
        
        # interpolate hw
        agent_bias2 = (self.ha_bias + self.wa_bias).squeeze(0).permute(0, 3, 1, 2)
        agent_bias2 = nn.functional.interpolate(agent_bias2, size=hw_shape, mode='bilinear')
        agent_bias2 = agent_bias2.reshape(num_heads, self.pool_size, self.pool_size, h * w).permute(0, 3, 1, 2)
        
        # interpolate agent_num
        agent_bias2 = nn.functional.interpolate(agent_bias2, size=self.downstream_agent_shape, mode='bilinear')
        agent_bias2 = agent_bias2.reshape(1, num_heads, h * w, downstream_agent_num).repeat(b, 1, 1, 1)
        agent_bias = agent_bias1 + agent_bias2
        q_attn = self.softmax((q * self.scale) @ agent_tokens.transpose(-2, -1) + agent_bias)
        q_attn = self.attn_drop(q_attn)
        x = q_attn @ agent_v

        x = x.transpose(1, 2).reshape(b, n, c)
        v = v.transpose(1, 2).reshape(b, h, w, c).permute(0, 3, 1, 2)
        x = x + self.dwc(v).permute(0, 2, 3, 1).reshape(b, n, c)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops

# 通过窗口的方式处理注意力计算
class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)
        print('Softmax Attention window{}'.format(window_size))

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)#线性变换生成Q K V
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        #相对位置偏置引入注意力矩阵
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:# not
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 agent_num=49, downstream_agent_shape=(7, 7), kernel_size=3, attn_type='B', scale=-0.5):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        # if min(self.input_resolution) <= self.window_size:
        #     # if window size is larger than input resolution, we don't partition windows
        #     self.shift_size = 0
        #     self.window_size = min(self.input_resolution)
        # assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim) 
        assert attn_type in ['B', 'A']
        self.attn_type = attn_type
        if attn_type == 'A':
            self.shift_size = 0
            self.attn = AgentAttention(dim=dim, input_resolution=input_resolution, window_size=to_2tuple(self.window_size), num_heads=num_heads,
                                       qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
                                       agent_num=agent_num, downstream_agent_shape=downstream_agent_shape,
                                       kernel_size=kernel_size, scale=scale)
        else:
            self.attn = WindowAttention(
                dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
                qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, hw_shape):
        shortcut = x
        x = self.norm1(x)#沿第三维度进行标准化处理
        
        B, L, C = x.shape
        H, W = hw_shape
        assert L == H * W, 'input feature has wrong size'
        x = x.view(B, H, W, C)# 改变张量形状

        # pad feature maps to multiples of window size
        if self.attn_type == 'B':
            pad_r = (self.window_size - W % self.window_size) % self.window_size#7
            pad_b = (self.window_size - H % self.window_size) % self.window_size
            x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))#计算需要在输入张量的右侧（pad_r）和底部（pad_b）分别添加的零填充数量
        H_pad, W_pad = x.shape[1], x.shape[2]

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(
                x,
                shifts=(-self.shift_size, -self.shift_size),#
                dims=(1, 2))#滚动将张量进行平移

            # calculate attention mask for SW-MSA
            img_mask = torch.zeros((1, H_pad, W_pad, 1), device=x.device)
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size,
                              -self.shift_size), slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size,
                              -self.shift_size), slice(-self.shift_size, None))
            cnt = 0
            #依次填充数值
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            # nW, window_size, window_size, 1
            mask_windows = window_partition(img_mask, self.window_size) # nW*B, window_size, window_size, C
            mask_windows = mask_windows.view(
                -1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0,
                                              float(-100.0)).masked_fill(
                                                  attn_mask == 0, float(0.0))
        else:
            shifted_x = x
            attn_mask = None

        if self.attn_type == 'B':
            # partition windows
            x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
            x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

            # W-MSA/SW-MSA
            attn_windows = self.attn(x_windows, attn_mask)  # nW*B, window_size*window_size, C

            # merge windows
            attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
            shifted_x = window_reverse(attn_windows, self.window_size, H_pad, W_pad)  # B H' W' C
        else:
            shifted_x = shifted_x.view(-1, H * W, C)
            # B, H*W, C in and B, H*W, C out
            shifted_x = self.attn(shifted_x, attn_mask, hw_shape)
            shifted_x = shifted_x.view(-1, H, W, C)

        # reverse cyclic shift
        if self.shift_size > 0:#0
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if self.attn_type == 'B' and (pad_r > 0 or pad_b > 0):
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, embed_dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        # self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)
        self.reduction = nn.Linear(4 * dim, embed_dim, bias=False)

    def forward(self, x, input_size):
        """
        x: B, H*W, C
        """
        H, W = input_size
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)

        # padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        output_size = (x.size(1), x.size(2))
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C
##############################################
        x = self.norm(x)
        x = self.reduction(x)#降维

        return x, output_size

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size, embed_dim,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 agent_num=49, downstream_agent_shape=(7, 7), kernel_size=3, attn_type='C', scale=-0.5):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        attn_types = [(attn_type if attn_type[0] != 'M' else ('A' if i < int(attn_type[1:]) else 'B')) for i in range(depth)]
        window_sizes = [(window_size if attn_types[i] == 'A' else max(7, (window_size // 8))) for i in range(depth)]
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_sizes[i],
                                 shift_size=0 if (i % 2 == 0) else window_sizes[i] // 2,#0 4 0 4
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer,
                                 agent_num=agent_num,
                                 downstream_agent_shape=downstream_agent_shape,
                                 kernel_size=kernel_size,
                                 attn_type=attn_types[i],
                                 scale=scale)
            for i in range(depth)])#2 2 6 2
      # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim,embed_dim=embed_dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, hw_shape):
        for blk in self.blocks:#2 2 6 2
            if self.use_checkpoint:# False
                x = checkpoint.checkpoint(blk, x, hw_shape)
            else:
                x = blk(x, hw_shape)
        if self.downsample:
            x_down, down_hw_shape = self.downsample(x, hw_shape)
            return x_down, down_hw_shape, x, hw_shape
        else:
            return x, hw_shape, x, hw_shape

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops

# 将输入图像转换为分块嵌入表示
class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        # padding
        _, _, H, W = x.shape
        if W % self.patch_size[1] != 0: #patch_size (4, 4)
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1]))
        if H % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))
        
          
        x = self.proj(x) #embed_dim = 96 
        out_size = (x.shape[2], x.shape[3])#H W
        x = x.flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None: # not None
            x = self.norm(x) #归一化
        return x, out_size

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops


@BACKBONES.register_module
class AgentSwinTransformer(nn.Module):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in _chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=56, mlp_ratio=4.,
                 out_indices=(0, 1, 2, 3),
                 qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False,
                 agent_num=[9, 16, 49, 49], downstream_agent_shapes=[(3, 3), (4, 4), (7, 7), (7, 7)],
                 kernel_size=3, attn_type='AAAA', scale=-0.5, 
                 init_cfg=None, 
                 embed=None,**kwargs):
        super().__init__()
        self.init_cfg = init_cfg
        self.attn_type = attn_type
        self.agent_num = agent_num
        self.downstream_agent_shapes = downstream_agent_shapes
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.out_indices = out_indices
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        # self.num_features = [int(embed_dim * 2**i) for i in range(self.num_layers)]#[64, 128, 256, 512]
        self.num_features=[embed[i] for i in range(self.num_layers)]

        # Add a norm layer for each output
        for i in out_indices:
            stages_norm_layer = norm_layer(self.num_features[i])
            stages_norm_layer_name = f'norm{i}'
            self.add_module(stages_norm_layer_name, stages_norm_layer)
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed[0],
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):#4
            layer = BasicLayer(#dim=int(embed_dim * 2 ** i_layer),##96 192 384 768
                               dim=embed[i_layer],
                               embed_dim=embed[i_layer+1 if i_layer!=self.num_layers-1 else i_layer-1] ,
                            #    input_resolution=(patches_resolution[0] // (2 ** i_layer),
                            #                      patches_resolution[1] // (2 ** i_layer)),
                               input_resolution=(embed[i_layer],
                                                 embed[i_layer]),
                               depth=depths[i_layer], 
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,#最后一次循环，x维度不变
                               use_checkpoint=use_checkpoint,
                               agent_num=int(agent_num[i_layer]),#[9, 16, 49, 49]
                               downstream_agent_shape = downstream_agent_shapes[i_layer],
                               kernel_size=kernel_size,
                               attn_type=attn_type[i_layer] + (attn_type[self.num_layers:] if attn_type[i_layer] == 'M' else ''),
                               scale=scale)
            self.layers.append(layer)
            conv_last=nn.Conv2d(embed[i_layer], 1, kernel_size=3, padding=1)
            setattr(self, f"conv_last{i_layer}", conv_last)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def init_weights(self,pretrained=None):
        logger = get_root_logger()
        assert 'checkpoint' in self.init_cfg, f'Only support ' \
                                                f'specify `Pretrained` in ' \
                                                f'`init_cfg` in ' \
                                                f'{self.__class__.__name__} '

        ckpt = _load_checkpoint(
            self.init_cfg['checkpoint'], logger=logger, map_location='cpu')
        if 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        elif 'model' in ckpt:
            state_dict = ckpt['model']
        else:
            state_dict = ckpt
        # load state_dict
        missing_keys, unexpected_keys = self.load_state_dict(state_dict, False)
        print('missing keys:', missing_keys)
        print('unexpected keys:', unexpected_keys)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward(self, x):
        if isinstance(x,list):
            x=x[0]
        x, hw_shape = self.patch_embed(x) #embed_dim 64

        if self.ape: #False
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        outs = []
        outs1=[]
        # 执行神经网络的前向传播过程，通过逐层遍历、计算、规范化和形状调整，生成神经网络的最终输出。
        for i, layer in enumerate(self.layers):
            x, hw_shape, out, out_hw_shape = layer(x, hw_shape)
            if i in self.out_indices:#0 1 2 3
                norm_layer = getattr(self, f'norm{i}')
                conv_last =getattr(self, f'conv_last{i}')
                out = norm_layer(out)
                out = out.view(-1, *out_hw_shape,
                               self.num_features[i]).permute(0, 3, 1,
                                                             2).contiguous()
                
                outs.append(out)
                outs1.append(conv_last(out))

        # return outs, outs[0], outs[1], outs[2], outs[3] 
        return outs, outs1[0], outs1[1], outs1[2], outs1[3]

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
        flops += self.num_features * self.num_classes
        return flops


class agent_swin_v1_b2(AgentSwinTransformer):
    def __init__(self, **kwargs):
        super(agent_swin_v1_b2, self).__init__(
        img_size=224, 
        patch_size=4, 
        in_chans=3,
        num_classes=80,
        # embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
 
        window_size=56,

        mlp_ratio=4,
        out_indices=(0, 1, 2, 3),
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.3,
        ape=False,
        patch_norm=True,
        use_checkpoint=False,
        agent_num=[9, 16, 49, 49],
        downstream_agent_shapes = [(9, 9), (12, 12), (14, 14), (7, 7)],
        kernel_size=3, 
        attn_type='AAAB',
        scale=-0.5,
        embed=[96, 192, 384, 768])

class agent_swin_v2_b2(AgentSwinTransformer):
    def __init__(self, **kwargs):
        super(agent_swin_v2_b2, self).__init__(
        img_size=224, 
        patch_size=4, 
        in_chans=3,
        num_classes=80,
        # embed_dim=96,
        depths=[2, 2, 6, 2],
        # num_heads=[3, 6, 12, 24],
        num_heads=[4, 8, 16, 32],
        # window_size=56,
        window_size=64,
        mlp_ratio=4,
        out_indices=(0, 1, 2, 3),
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.3,
        ape=False,
        patch_norm=True,
        use_checkpoint=False,
        agent_num=[9, 16, 49, 49],
        downstream_agent_shapes = [(9, 9), (12, 12), (14, 14), (7, 7)],
        kernel_size=3, 
        attn_type='AAAB',
        scale=-0.5,
        embed=[64,128, 320, 512])

if __name__=='__main__':
    x=torch.rand(36,3,224, 224)
    # model = AgentSwinTransformer()#img_size=256, embed_dim=64, window_size=64,num_heads=(1,2,4,8))
    # module=AgentSwinTransformer()
    model=agent_swin_v2_b2()
    y=model(x)
    print('sucess')