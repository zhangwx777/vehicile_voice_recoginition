#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CTFA (Channel-Time-Frequency Attention) 模块
用于通道-时间-频率三维注意力机制，实现干扰抑制和特征增强
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class CTFA(nn.Module):
    """Channel-Time-Frequency Attention 模块"""
    
    def __init__(self, in_channels, reduction_ratio=16, 
                 time_kernel_size=7, freq_kernel_size=7):
        """
        初始化CTFA模块
        
        Args:
            in_channels: 输入通道数
            reduction_ratio: 降维比例
            time_kernel_size: 时间维度卷积核大小
            freq_kernel_size: 频率维度卷积核大小
        """
        super(CTFA, self).__init__()
        
        self.in_channels = in_channels
        self.reduction_ratio = reduction_ratio
        
        # 通道注意力分支
        self.channel_attention = ChannelAttention3D(in_channels, reduction_ratio)
        
        # 时间注意力分支
        self.time_attention = TemporalAttention(in_channels, time_kernel_size)
        
        # 频率注意力分支
        self.frequency_attention = FrequencyAttention(in_channels, freq_kernel_size)
        
        # 特征融合卷积
        self.fusion_conv = nn.Conv2d(in_channels * 3, in_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入特征图 [batch, channels, freq, time]
        
        Returns:
            增强后的特征图
        """
        # 通道注意力
        channel_att = self.channel_attention(x)
        channel_out = channel_att * x
        
        # 时间注意力
        time_att = self.time_attention(x)
        time_out = time_att * x
        
        # 频率注意力
        freq_att = self.frequency_attention(x)
        freq_out = freq_att * x
        
        # 拼接三个注意力分支的输出
        combined = torch.cat([channel_out, time_out, freq_out], dim=1)
        
        # 特征融合
        out = self.fusion_conv(combined)
        out = self.bn(out)
        out = self.relu(out)
        
        # 残差连接
        out = out + x
        
        return out

class ChannelAttention3D(nn.Module):
    """三维通道注意力机制"""
    
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention3D, self).__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # 共享的MLP
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels)
        )
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        b, c, _, _ = x.size()
        
        # 平均池化路径
        avg_out = self.mlp(self.avg_pool(x).view(b, c))
        # 最大池化路径
        max_out = self.mlp(self.max_pool(x).view(b, c))
        
        # 合并注意力权重
        out = avg_out + max_out
        out = self.sigmoid(out).view(b, c, 1, 1)
        
        return out

class TemporalAttention(nn.Module):
    """时间维度注意力机制"""
    
    def __init__(self, in_channels, kernel_size=7):
        super(TemporalAttention, self).__init__()
        
        assert kernel_size % 2 == 1, "Kernel size must be odd"
        padding = kernel_size // 2
        
        # 时间卷积层
        self.conv = nn.Conv1d(in_channels, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # x shape: [batch, channels, freq, time]
        b, c, f, t = x.size()
        
        # 沿频率维度平均池化
        time_feat = torch.mean(x, dim=2)  # [batch, channels, time]
        
        # 时间注意力计算
        time_att = self.conv(time_feat)  # [batch, 1, time]
        time_att = self.sigmoid(time_att)
        
        # 扩展维度以匹配输入
        time_att = time_att.unsqueeze(2)  # [batch, 1, 1, time]
        
        return time_att

class FrequencyAttention(nn.Module):
    """频率维度注意力机制"""
    
    def __init__(self, in_channels, kernel_size=7):
        super(FrequencyAttention, self).__init__()
        
        assert kernel_size % 2 == 1, "Kernel size must be odd"
        padding = kernel_size // 2
        
        # 频率卷积层
        self.conv = nn.Conv1d(in_channels, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # x shape: [batch, channels, freq, time]
        b, c, f, t = x.size()
        
        # 沿时间维度平均池化
        freq_feat = torch.mean(x, dim=3)  # [batch, channels, freq]
        
        # 频率注意力计算
        freq_att = self.conv(freq_feat)  # [batch, 1, freq]
        freq_att = self.sigmoid(freq_att)
        
        # 扩展维度以匹配输入
        freq_att = freq_att.unsqueeze(3)  # [batch, 1, freq, 1]
        
        return freq_att

class MultiHeadCTFA(nn.Module):
    """多头CTFA模块，用于捕获不同表示子空间的注意力"""
    
    def __init__(self, in_channels, num_heads=8, reduction_ratio=16):
        super(MultiHeadCTFA, self).__init__()
        
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads
        
        assert self.head_dim * num_heads == in_channels, "in_channels must be divisible by num_heads"
        
        # 创建多个CTFA头
        self.heads = nn.ModuleList([
            CTFA(self.head_dim, reduction_ratio) for _ in range(num_heads)
        ])
        
        # 输出投影
        self.proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(in_channels)
    
    def forward(self, x):
        b, c, f, t = x.size()
        
        # 分割特征到多个头
        x = x.view(b, self.num_heads, self.head_dim, f, t)
        
        # 每个头独立处理
        outputs = []
        for i in range(self.num_heads):
            head_x = x[:, i, :, :, :]
            head_out = self.heads[i](head_x)
            outputs.append(head_out)
        
        # 合并多头输出
        out = torch.cat(outputs, dim=1)
        out = out.view(b, c, f, t)
        
        # 输出投影
        out = self.proj(out)
        out = self.bn(out)
        
        # 残差连接
        out = out + x.view(b, c, f, t)
        
        return out

class CTFANetwork(nn.Module):
    """CTFA网络，包含多个CTFA模块的堆叠"""
    
    def __init__(self, in_channels, num_blocks=3, num_heads=8):
        super(CTFANetwork, self).__init__()
        
        self.blocks = nn.ModuleList([
            MultiHeadCTFA(in_channels, num_heads) for _ in range(num_blocks)
        ])
        
        # 最终输出卷积
        self.final_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        # 通过多个CTFA块
        for block in self.blocks:
            x = block(x)
        
        # 最终卷积
        out = self.final_conv(x)
        out = self.bn(out)
        out = self.relu(out)
        
        return out

def test_ctfa():
    """测试CTFA模块"""
    print("=== 测试CTFA模块 ===")
    
    # 创建测试输入 (模拟mel频谱图)
    batch_size, channels, freq_bins, time_steps = 4, 64, 128, 100
    x = torch.randn(batch_size, channels, freq_bins, time_steps)
    
    # 测试基础CTFA
    ctfa = CTFA(in_channels=64)
    out = ctfa(x)
    print(f"输入形状: {x.shape}")
    print(f"CTFA输出形状: {out.shape}")
    
    # 测试多头CTFA
    multihead_ctfa = MultiHeadCTFA(in_channels=64, num_heads=8)
    out_mh = multihead_ctfa(x)
    print(f"多头CTFA输出形状: {out_mh.shape}")
    
    # 测试CTFA网络
    ctfa_network = CTFANetwork(in_channels=64, num_blocks=3)
    out_net = ctfa_network(x)
    print(f"CTFA网络输出形状: {out_net.shape}")
    
    print("CTFA模块测试完成!")

if __name__ == "__main__":
    test_ctfa()
