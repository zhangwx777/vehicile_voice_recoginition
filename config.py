#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
车辆声纹识别系统统一配置文件
整合原系统和增强系统的所有配置选项
"""

import os
from pathlib import Path

# 基础路径配置
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "vehicle_audio_data"
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
    
    # 具体文件路径
    'individual_data_dir': str(DATA_DIR / 'individual_recognition'),
    'vector_database_path': str(MODELS_DIR / 'similarity_database.json'),
    'similarity_index_path': str(MODELS_DIR / 'similarity_index.pkl'),
    'cnn_model_path': str(MODELS_DIR / 'individual_best_model.pth'),
    'label_mapping_path': str(MODELS_DIR / 'individual_label_mapping.json'),
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

# CNN模型配置
CNN_MODEL_CONFIG = {
    'input_channels': 1,
    'base_filters': 16,
    'num_conv_blocks': 3,
    'dropout_rate': 0.3,
    'num_classes': 25,  # 25个车辆个体
}

# 增强特征提取器配置
ENHANCED_FEATURE_CONFIG = {
    'target_sr': 16000,
    'feature_dim': 512,
    'use_pca': True,
    'pca_components': 256,
    'normalize_features': True,
    'device': 'auto',
    
    # 特征融合权重
    'fusion_weights': {
        'yamnet': 0.6,
        'vggish': 0.4
    }
}

# 向量相似度搜索配置
SIMILARITY_CONFIG = {
    'similarity_threshold': 0.7,
    'max_candidates': 10,
    'distance_metric': 'cosine',
    'index_type': 'flat',  # 可选: 'flat', 'ivf', 'hnsw'
    
    # 搜索参数
    'search_params': {
        'nprobe': 10,  # IVF索引参数
        'ef': 50,      # HNSW索引参数
    }
}

# 训练配置
TRAINING_CONFIG = {
    'batch_size': 8,
    'num_epochs': 100,
    'learning_rate': 0.0005,
    'weight_decay': 1e-3,
    'step_size': 30,
    'gamma': 0.5,
    
    # 早停配置
    'early_stopping': {
        'patience': 25,
        'min_delta': 0.001,
        'restore_best_weights': True,
    },
    
    # 梯度裁剪
    'gradient_clipping': {
        'enabled': True,
        'max_norm': 0.5,
    },
    
    # 学习率调度
    'scheduler': {
        'type': 'StepLR',  # 可选: 'StepLR', 'CosineAnnealingLR', 'ReduceLROnPlateau'
        'step_size': 30,
        'gamma': 0.5,
    }
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
    
    # 分类模式配置（已禁用）
    'classification_mode': {
        'enabled': False,  # 禁用分类功能
        'model_type': 'cnn',
        'confidence_threshold': 0.8,
    },
    
    # 相似度搜索模式配置
    'similarity_mode': {
        'enabled': True,
        'feature_extractor': 'enhanced',
        'similarity_threshold': 0.7,
        'max_results': 10,
    },
    
    # 混合模式配置（已禁用）
    'hybrid_mode': {
        'enabled': False,  # 禁用混合模式
        'primary_method': 'similarity',
        'fallback_method': 'classification',
        'confidence_threshold': 0.6,
        'similarity_threshold': 0.5,
    }
}

# API配置已删除 - 仅本地使用模式

# 日志配置
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file_enabled': True,
    'file_path': 'logs/system.log',
    'max_file_size': 10 * 1024 * 1024,  # 10MB
    'backup_count': 5,
    'console_enabled': True
}

# 性能监控配置
MONITORING_CONFIG = {
    'enabled': True,
    'metrics': {
        'inference_time': True,
        'memory_usage': True,
        'cpu_usage': True,
        'gpu_usage': True,
    },
    'alerts': {
        'max_inference_time': 5.0,  # 秒
        'max_memory_usage': 80,     # 百分比
        'max_cpu_usage': 90,        # 百分比
    }
}

# 缓存配置
CACHE_CONFIG = {
    'enabled': True,
    'type': 'memory',  # 可选: 'memory', 'redis', 'file'
    'ttl': 3600,       # 缓存过期时间（秒）
    'max_size': 1000,  # 最大缓存条目数
    
    # Redis配置（如果使用Redis缓存）
    'redis': {
        'host': 'localhost',
        'port': 6379,
        'db': 0,
        'password': None,
    }
}

# 导出所有配置
__all__ = [
    'PATH_CONFIG',
    'AUDIO_CONFIG', 
    'CNN_MODEL_CONFIG',
    'ENHANCED_FEATURE_CONFIG',
    'SIMILARITY_CONFIG',
    'TRAINING_CONFIG',
    'DATA_CONFIG',
    'SYSTEM_MODE_CONFIG',
    'LOGGING_CONFIG',
    'MONITORING_CONFIG',
    'CACHE_CONFIG'
]