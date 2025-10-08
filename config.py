#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
车辆声纹识别系统统一配置文件
整合原系统和增强系统的所有配置选项
"""

# 直接导入避免循环依赖
import os
import sys
import time
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
import warnings
import numpy as np

# 初始化日志器 - 延迟导入
def get_config_logger():
    from core.logger import ComponentLogger
    return ComponentLogger("配置管理")

# 基础路径配置
BASE_DIR = Path(__file__).parent
# 数据目录配置 - 使用enhanced_virtual_audio作为主要音频数据目录
DATA_DIR = BASE_DIR / "enhanced_virtual_audio"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
LOGS_DIR = BASE_DIR / "logs"
CHECKPOINTS_DIR = BASE_DIR / "checkpoints"

# 确保目录存在
for dir_path in [DATA_DIR, MODELS_DIR, RESULTS_DIR, LOGS_DIR, CHECKPOINTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# 路径配置
PATH_CONFIG = {
    'base_dir': str(BASE_DIR),
    'data_dir': str(DATA_DIR),
    'models_dir': str(MODELS_DIR),
    'results_dir': str(RESULTS_DIR),
    'logs_dir': str(LOGS_DIR),
    'checkpoints_dir': str(CHECKPOINTS_DIR),
    'pretrained_models_dir': str(BASE_DIR / 'pretrained_models'),
    
    # 具体文件路径
    'individual_data_dir': str(DATA_DIR / 'individual_recognition'),
    'vector_database_path': str(MODELS_DIR / 'ecapa_similarity_database.json'),
    'ecapa_model_path': str(BASE_DIR / 'pretrained_models' / 'ecapa_tdnn'),
}

# 音频处理配置
AUDIO_CONFIG = {
    'sample_rate': 16000,
    'n_mels': 128,
    'n_fft': 2048,
    'hop_length': 512,
    'win_length': 2048,
    'fmin': 0,
    'fmax': 8000,
    'duration': 3.0,  # 音频时长（秒）
    'normalize': True,
    'to_db': True,
    'ref': 1.0,
    'amin': 1e-10,
    'top_db': 80.0,
    
    # 数据增强配置
    'augmentation': {
        'time_stretch': {
            'enabled': True,
            'rate_range': [0.8, 1.2],
            'probability': 0.3
        },
        'pitch_shift': {
            'enabled': True,
            'steps_range': [-2, 2],
            'probability': 0.3
        },
        'noise_injection': {
            'enabled': True,
            'noise_factor': 0.005,
            'probability': 0.2
        },
        'time_mask': {
            'enabled': True,
            'max_mask_pct': 0.1,
            'num_masks': 2,
            'probability': 0.3
        },
        'freq_mask': {
            'enabled': True,
            'max_mask_pct': 0.1,
            'num_masks': 2,
            'probability': 0.3
        }
    }
}





# ECAPA-TDNN特征提取器配置
ECAPA_FEATURE_CONFIG = {
    'use_cache': True,
    'max_workers': 2
}

# 相似度搜索配置
SIMILARITY_CONFIG = {
    'similarity_threshold': 0.7,  # 恢复正常的相似度阈值 (0-1范围)
    'max_candidates': 10,
    'feature_dim': 192,
    'index_type': 'auto',
    'use_gpu': False,
    'n_probe': 10
}



# 数据配置
DATA_CONFIG = {
    'max_samples_per_class': 500,
    'train_split': 0.7,
    'val_split': 0.15,
    'test_split': 0.15,
    'stratify': True,
    'shuffle': True,
    'num_workers': 4,
    'pin_memory': True,
    'enable_cache': False,  # 是否启用特征缓存
}

# 系统模式配置
SYSTEM_MODE_CONFIG = {
    'default_mode': 'similarity',  # 只使用相似度搜索模式
    
    # 相似度搜索模式配置
    'similarity_mode': {
        'enabled': True,
        'feature_extractor': 'ecapa',  # 使用ECAPA-TDNN特征提取器
        'similarity_threshold': 0.7,  # 恢复正常的相似度阈值 (0-1范围)
        'max_results': 10,
    }
}

# 日志配置
LOGGING_CONFIG = {
    'level': 'INFO',  # 恢复正常的日志级别
    'format': '%(asctime)s | %(levelname)-8s | %(name)-25s | %(message)s',
    'file_enabled': True,
    'file_path': str(LOGS_DIR / 'system.log'),
    'max_file_size': 10 * 1024 * 1024,  # 10MB
    'console_enabled': True
}

# 导出所有配置
__all__ = [
    'PATH_CONFIG',
    'AUDIO_CONFIG', 
    'ECAPA_FEATURE_CONFIG',
    'SIMILARITY_CONFIG',
    'DATA_CONFIG',
    'SYSTEM_MODE_CONFIG',
    'LOGGING_CONFIG'
]