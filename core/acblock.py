#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ACBlock (Attention Convolution Block) 模块
用于网络层的特征筛选和干扰抑制
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ACBlock(nn.Module):
    """Attention Convolution Block - 注意力卷积块"""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                 dilation=1, groups=1, bias=True, reduction_ratio=16):
        """
        初始化ACBlock
        
        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数
            kernel_size: 卷积核大小
            stride: 步长
            padding: 填充
            dilation: 膨胀率
            groups: 分组卷积
            bias: 是否使用偏置
            reduction_ratio: 通道注意力中的降维比例
        """
        super(ACBlock, self).__init__()
        
        # 主卷积路径
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, 
            padding, dilation, groups, bias
        )
        
        # 通道注意力机制
        self.channel_attention = ChannelAttention(out_channels, reduction_ratio)
        
        # 空间注意力机制
        self.spatial_attention = SpatialAttention()
        
        # 批归一化
        self.bn = nn.BatchNorm2d(out_channels)
        
        # 激活函数
        self.activation = nn.ReLU(inplace=True)
        
        # 残差连接（如果输入输出维度匹配）
        self.use_residual = (in_channels == out_channels and stride == 1)
        
    def forward(self, x):
        """前向传播"""
        residual = x
        
        # 主卷积路径
        out = self.conv(x)
        out = self.bn(out)
        
        # 通道注意力
        out = self.channel_attention(out) * out
        
        # 空间注意力
        out = self.spatial_attention(out) * out
        
        # 激活函数
        out = self.activation(out)
        
        # 残差连接
        if self.use_residual:
            out = out + residual
        
        return out

class ChannelAttention(nn.Module):
    """通道注意力机制"""
    
    def __init__(self, channel, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction_ratio, channel, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        b, c, _, _ = x.size()
        
        # 平均池化路径
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        # 最大池化路径
        max_out = self.fc(self.max_pool(x).view(b, c))
        
        # 合并注意力权重
        out = avg_out + max_out
        out = self.sigmoid(out).view(b, c, 1, 1)
        
        return out

class SpatialAttention(nn.Module):
    """空间注意力机制"""
    
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # 沿通道维度计算平均值和最大值
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        # 拼接特征
        out = torch.cat([avg_out, max_out], dim=1)
        
        # 卷积处理
        out = self.conv(out)
        
        # 生成空间注意力权重
        out = self.sigmoid(out)
        
        return out

class MultiScaleACBlock(nn.Module):
    """多尺度ACBlock，用于处理不同尺度的特征"""
    
    def __init__(self, in_channels, out_channels):
        super(MultiScaleACBlock, self).__init__()
        
        # 不同尺度的卷积路径
        self.conv3x3 = ACBlock(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv5x5 = ACBlock(in_channels, out_channels, kernel_size=5, padding=2)
        self.conv7x7 = ACBlock(in_channels, out_channels, kernel_size=7, padding=3)
        
        # 特征融合
        self.fusion_conv = nn.Conv2d(out_channels * 3, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        # 多尺度特征提取
        out3x3 = self.conv3x3(x)
        out5x5 = self.conv5x5(x)
        out7x7 = self.conv7x7(x)
        
        # 拼接多尺度特征
        out = torch.cat([out3x3, out5x5, out7x7], dim=1)
        
        # 特征融合
        out = self.fusion_conv(out)
        out = self.bn(out)
        out = self.relu(out)
        
        return out

def test_acblock():
    """测试ACBlock模块"""
    print("=== 测试ACBlock模块 ===")
    
    # 创建测试输入
    batch_size, channels, height, width = 4, 64, 64, 64
    x = torch.randn(batch_size, channels, height, width)
    
    # 测试基础ACBlock
    acblock = ACBlock(in_channels=64, out_channels=64)
    out = acblock(x)
    print(f"输入形状: {x.shape}")
    print(f"ACBlock输出形状: {out.shape}")
    
    # 测试多尺度ACBlock
    ms_acblock = MultiScaleACBlock(in_channels=64, out_channels=64)
    out_ms = ms_acblock(x)
    print(f"多尺度ACBlock输出形状: {out_ms.shape}")
    
    print("ACBlock模块测试完成!")

if __name__ == "__main__":
    test_acblock()
