#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
车辆音频数据加载器模块
包含基础数据集类和相关工具函数
"""

import os
import gc
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import logging
from core.logger import logger
from core.settings import PATH_CONFIG, AUDIO_CONFIG
from data.preprocessor import AudioPreprocessor

logger = logging.getLogger(__name__)

class VehicleSubDataset(Dataset):
    """车辆音频子数据集类
    
    用于创建训练、验证和测试数据集的子集
    """
    
    def __init__(self, data_dir, preprocessor, file_paths, labels, label_to_idx, idx_to_label, enable_cache=False):
        """
        初始化子数据集
        
        Args:
            data_dir: 数据目录路径
            preprocessor: 音频预处理器
            file_paths: 文件路径列表
            labels: 标签列表
            label_to_idx: 标签到索引的映射
            idx_to_label: 索引到标签的映射
            enable_cache: 是否启用缓存
        """
        self.data_dir = data_dir
        self.preprocessor = preprocessor
        self.file_paths = file_paths
        self.labels = labels
        self.label_to_idx = label_to_idx
        self.idx_to_label = idx_to_label
        self.feature_cache = {} if enable_cache else None
        
        logger.info(f"创建子数据集: {len(self.file_paths)} 个样本")
    
    def __len__(self):
        """返回数据集大小"""
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        """获取单个样本
        
        Args:
            idx: 样本索引
            
        Returns:
            tuple: (features, label) 特征和标签
        """
        try:
            file_path = self.file_paths[idx]
            label = self.labels[idx]
            
            # 检查缓存
            if self.feature_cache is not None and file_path in self.feature_cache:
                features = self.feature_cache[file_path]
            else:
                # 使用预处理器提取特征
                features = self.preprocessor.preprocess_audio(file_path)
                
                # 缓存特征
                if self.feature_cache is not None:
                    self.feature_cache[file_path] = features
            
            # 确保特征是正确的格式
            if isinstance(features, dict):
                # 如果返回多个特征，使用mel频谱图
                features = features.get('mel', features.get('mfcc', list(features.values())[0]))
            
            # 转换为张量
            if not isinstance(features, torch.Tensor):
                features = torch.tensor(features, dtype=torch.float32)
            
            # 确保有正确的维度
            if features.dim() == 2:
                features = features.unsqueeze(0)  # 添加通道维度
            
            # 确保标签是整数
            if isinstance(label, str):
                label = self.label_to_idx.get(label, 0)
            
            return features, label
            
        except Exception as e:
            logger.error(f"加载样本失败 {idx}: {file_path} - {str(e)}")
            # 返回零特征和标签
            dummy_features = torch.zeros((128, 87), dtype=torch.float32)  # 默认MFCC特征维度
            return dummy_features, 0
    
    def get_label_name(self, idx):
        """根据索引获取标签名称"""
        return self.idx_to_label.get(idx, f"unknown_{idx}")
    
    def clear_cache(self):
        """清理特征缓存"""
        if self.feature_cache:
            self.feature_cache.clear()
            gc.collect()
            logger.info("特征缓存已清理")


def create_data_loaders(data_dir, batch_size=32, test_size=0.2, val_size=0.1, random_state=42):
    """创建数据加载器
    
    Args:
        data_dir: 数据目录路径
        batch_size: 批次大小
        test_size: 测试集比例
        val_size: 验证集比例
        random_state: 随机种子
    
    Returns:
        tuple: (train_loader, val_loader, test_loader, label_to_idx, idx_to_label)
    """
    try:
        # 初始化预处理器
        preprocessor = AudioPreprocessor()
        
        # 收集所有音频文件
        file_paths = []
        labels = []
        
        for label_dir in os.listdir(data_dir):
            label_path = os.path.join(data_dir, label_dir)
            if os.path.isdir(label_path):
                for file_name in os.listdir(label_path):
                    if file_name.endswith(('.wav', '.mp3', '.flac')):
                        file_paths.append(os.path.join(label_path, file_name))
                        labels.append(label_dir)
        
        # 创建标签映射
        unique_labels = sorted(list(set(labels)))
        label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        idx_to_label = {idx: label for label, idx in label_to_idx.items()}
        
        # 分割数据集
        train_files, test_files, train_labels, test_labels = train_test_split(
            file_paths, labels, test_size=test_size, random_state=random_state, stratify=labels
        )
        
        train_files, val_files, train_labels, val_labels = train_test_split(
            train_files, train_labels, test_size=val_size, random_state=random_state, stratify=train_labels
        )
        
        # 创建数据集
        train_dataset = VehicleSubDataset(data_dir, preprocessor, train_files, train_labels, label_to_idx, idx_to_label)
        val_dataset = VehicleSubDataset(data_dir, preprocessor, val_files, val_labels, label_to_idx, idx_to_label)
        test_dataset = VehicleSubDataset(data_dir, preprocessor, test_files, test_labels, label_to_idx, idx_to_label)
        
        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        logger.info(f"数据加载器创建成功: 训练集{len(train_dataset)}, 验证集{len(val_dataset)}, 测试集{len(test_dataset)}")
        
        return train_loader, val_loader, test_loader, label_to_idx, idx_to_label
        
    except Exception as e:
        logger.error(f"创建数据加载器失败: {str(e)}")
        raise