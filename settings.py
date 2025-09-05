# config/settings.py

import os
import torch
from pathlib import Path

# 获取项目根目录 - 使用更可靠的方法
try:
    # 首先尝试从环境变量获取
    PROJECT_ROOT = os.environ.get('VEHICLE_RECOGNITION_ROOT')
    if not PROJECT_ROOT:
        # 如果环境变量不存在，使用当前文件路径计算
        PROJECT_ROOT = str(Path(__file__).parent.absolute())
except Exception:
    # 后备方案
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# 设备配置
DEVICE_CONFIG = {
    'use_cuda': torch.cuda.is_available(),
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'gpu_memory_fraction': 0.8,  # GPU内存使用比例
    'mixed_precision': True,  # 混合精度训练
    'deterministic': True,  # 确定性计算
}

# 音频处理参数
AUDIO_CONFIG = {
    'sample_rate': 16000,
    'n_fft': 2048,
    'hop_length': 512,
    'n_mels': 128,
    'duration': 3,  # 音频时长(秒)
    'enable_augmentation': True,
    'augmentation_config': {
        'time_stretch': {
            'enabled': True,
            'rate_range': (0.8, 1.2),
            'probability': 0.3
        },
        'pitch_shift': {
            'enabled': True,
            'steps_range': (-2, 2),
            'probability': 0.3
        },
        'noise_injection': {
            'enabled': True,
            'noise_factor_range': (0.01, 0.05),
            'probability': 0.4
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
    },
    'extract_features': ['mel']
}

# 训练参数
TRAINING_CONFIG = {
    'batch_size': 8,  # 减小batch size，增加训练难度
    'num_epochs': 100,  # 增加epoch数
    'learning_rate': 0.0005,  # 降低学习率
    'weight_decay': 1e-3,  # 增加权重衰减
    'step_size': 30,
    'gamma': 0.5,  # 更激进的学习率衰减
    'early_stopping': {
        'patience': 25,  # 增加耐心值
        'min_delta': 0.001,  # 保持原有阈值
        'restore_best_weights': True,
    },
    'gradient_clipping': {
        'enabled': True,
        'max_norm': 0.5,  # 更严格的梯度裁剪
    }
}

# 模型参数
MODEL_CONFIG = {
    'input_channels': 1,
    'base_filters': 16,  # 减少基础滤波器数量
    'num_conv_blocks': 3,  # 减少卷积块数量
    'dropout_rate': 0.3,  # 降低dropout率
}

# 数据配置
DATA_CONFIG = {
    'max_samples_per_class': 500,  # 增加样本限制
    'train_split': 0.7,
    'val_split': 0.15,
    'test_split': 0.15,
    'stratify': True,  # 分层采样
    'shuffle': True,
    'num_workers': 4,  # 数据加载器工作进程数
    'pin_memory': True,  # 是否固定内存
}

# 路径配置
PATH_CONFIG = {
    'data_dir': os.path.join(PROJECT_ROOT, 'vehicle_audio_data'),
    'model_save_path': os.path.join(PROJECT_ROOT, 'models', 'best_model_epoch_100.pth'),
    'label_mapping_path': os.path.join(PROJECT_ROOT, 'models', 'label_mapping.json'),
    'results_dir': os.path.join(PROJECT_ROOT, 'results'),
    'logs_dir': os.path.join(PROJECT_ROOT, 'logs'),
    'checkpoints_dir': os.path.join(PROJECT_ROOT, 'checkpoints'),
}

# 日志配置
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file_rotation': True,
    'max_file_size': '10MB',
    'backup_count': 5,
}

# 导入配置管理器
try:
    from config_manager import ConfigManager
    
    # 自动验证配置
    if __name__ != '__main__':
        config_manager = ConfigManager()
        config_manager.validate_config()
except ImportError:
    # 如果配置管理器导入失败，使用基本验证
    def basic_config_validation():
        """基础配置验证，作为配置管理器的后备方案"""
        errors = []
        
        # 验证关键配置参数
        if AUDIO_CONFIG.get('sample_rate', 0) <= 0:
            errors.append("采样率必须大于0")
        if AUDIO_CONFIG.get('duration', 0) <= 0:
            errors.append("音频时长必须大于0")
        if TRAINING_CONFIG.get('batch_size', 0) <= 0:
            errors.append("批次大小必须大于0")
        
        # 创建必要的目录
        for key, path in PATH_CONFIG.items():
            if key.endswith('_dir'):
                os.makedirs(path, exist_ok=True)
        
        if errors:
            raise ValueError(f"基础配置验证失败: {'; '.join(errors)}")
        
    # 执行基础验证
    if __name__ != '__main__':
        basic_config_validation()