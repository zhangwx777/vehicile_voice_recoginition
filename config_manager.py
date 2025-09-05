# scripts/config_manager.py

import os
import json
import torch
from pathlib import Path
from logger import logger
from settings import *

class ConfigManager:
    """配置管理器"""
    
    def __init__(self):
        self.configs = {
            'device': DEVICE_CONFIG,
            'audio': AUDIO_CONFIG,
            'training': TRAINING_CONFIG,
            'model': MODEL_CONFIG,
            'data': DATA_CONFIG,
            'path': PATH_CONFIG,
            'logging': LOGGING_CONFIG
        }
    
    def validate_config(self):
        """验证配置"""
        logger.info("🔍 正在验证系统配置...")
        
        errors = []
        warnings = []
        
        # 验证设备配置
        try:
            if DEVICE_CONFIG['use_cuda'] and not torch.cuda.is_available():
                warnings.append("配置要求使用CUDA，但CUDA不可用，将使用CPU")
                
            gpu_memory = DEVICE_CONFIG.get('gpu_memory_fraction', 0.8)
            if gpu_memory <= 0 or gpu_memory > 1:
                errors.append(f"GPU内存使用比例无效: {gpu_memory}")
                
        except Exception as e:
            errors.append(f"设备配置验证失败: {str(e)}")
        
        # 验证音频配置
        try:
            audio_cfg = AUDIO_CONFIG
            if audio_cfg['sample_rate'] <= 0:
                errors.append("采样率必须大于0")
            if audio_cfg['duration'] <= 0:
                errors.append("音频时长必须大于0")
            if audio_cfg['n_mels'] <= 0:
                errors.append("梅尔频谱维度必须大于0")
                
        except Exception as e:
            errors.append(f"音频配置验证失败: {str(e)}")
        
        # 验证训练配置
        try:
            training_cfg = TRAINING_CONFIG
            if training_cfg['batch_size'] <= 0:
                errors.append("批次大小必须大于0")
            if training_cfg['learning_rate'] <= 0:
                errors.append("学习率必须大于0")
            if training_cfg['num_epochs'] <= 0:
                errors.append("训练轮次必须大于0")
                
        except Exception as e:
            errors.append(f"训练配置验证失败: {str(e)}")
        
        # 验证路径配置
        try:
            for key, path in PATH_CONFIG.items():
                if key.endswith('_dir'):
                    Path(path).mkdir(parents=True, exist_ok=True)
                    logger.debug(f"确保目录存在: {path}")
                    
        except Exception as e:
            errors.append(f"路径配置验证失败: {str(e)}")
        
        # 输出验证结果
        if errors:
            logger.error("❌ 配置验证失败:")
            for error in errors:
                logger.error(f"  - {error}")
            return False
        
        if warnings:
            logger.warning("⚠️  配置警告:")
            for warning in warnings:
                logger.warning(f"  - {warning}")
        
        logger.info("✅ 配置验证通过")
        return True
    
    def print_config_summary(self):
        """打印配置摘要"""
        logger.info("\n📋 系统配置摘要:")
        logger.info("=" * 60)
        
        # 设备信息
        logger.info(f"🖥️  设备配置:")
        logger.info(f"   使用设备: {DEVICE_CONFIG['device']}")
        logger.info(f"   混合精度: {DEVICE_CONFIG['mixed_precision']}")
        logger.info(f"   GPU内存比例: {DEVICE_CONFIG.get('gpu_memory_fraction', 0.8)}")
        
        # 音频参数
        logger.info(f"\n🎵 音频配置:")
        logger.info(f"   采样率: {AUDIO_CONFIG['sample_rate']} Hz")
        logger.info(f"   音频时长: {AUDIO_CONFIG['duration']} 秒")
        logger.info(f"   梅尔频谱维度: {AUDIO_CONFIG['n_mels']}")
        logger.info(f"   数据增强: {'启用' if AUDIO_CONFIG.get('augmentation_config', {}) else '禁用'}")
        
        # 训练参数
        logger.info(f"\n🏋️  训练配置:")
        logger.info(f"   批次大小: {TRAINING_CONFIG['batch_size']}")
        logger.info(f"   训练轮次: {TRAINING_CONFIG['num_epochs']}")
        logger.info(f"   学习率: {TRAINING_CONFIG['learning_rate']}")
        logger.info(f"   早停耐心值: {TRAINING_CONFIG.get('early_stopping', {}).get('patience', 10)}")
        
        # 模型参数
        logger.info(f"\n🧠 模型配置:")
        logger.info(f"   卷积块数: {MODEL_CONFIG['num_conv_blocks']}")
        logger.info(f"   基础滤波器: {MODEL_CONFIG['base_filters']}")
        logger.info(f"   Dropout率: {MODEL_CONFIG['dropout_rate']}")
        
        # 数据配置
        logger.info(f"\n📊 数据配置:")
        logger.info(f"   每类最大样本数: {DATA_CONFIG['max_samples_per_class']}")
        logger.info(f"   训练集比例: {DATA_CONFIG['train_split']}")
        logger.info(f"   验证集比例: {DATA_CONFIG['val_split']}")
        logger.info(f"   测试集比例: {DATA_CONFIG['test_split']}")
        
        logger.info("=" * 60)
    
    def save_config(self, save_path=None):
        """保存当前配置"""
        if save_path is None:
            save_path = Path(PATH_CONFIG['results_dir']) / 'training_config.json'
        
        try:
            # 准备可序列化的配置
            serializable_config = {}
            for key, config in self.configs.items():
                serializable_config[key] = dict(config)
            
            # 添加运行时信息
            serializable_config['runtime_info'] = {
                'torch_version': torch.__version__,
                'cuda_available': torch.cuda.is_available(),
                'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
            }
            
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(serializable_config, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✅ 配置已保存至: {save_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ 保存配置失败: {str(e)}")
            return False
    
    def optimize_for_hardware(self):
        """根据硬件自动优化配置"""
        logger.info("🔧 正在根据硬件优化配置...")
        
        # GPU优化
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"检测到GPU内存: {gpu_memory:.1f}GB")
            
            # 根据显存调整批次大小
            if gpu_memory >= 8:
                recommended_batch_size = 32
            elif gpu_memory >= 4:
                recommended_batch_size = 16
            else:
                recommended_batch_size = 8
            
            if TRAINING_CONFIG['batch_size'] != recommended_batch_size:
                logger.info(f"建议批次大小: {recommended_batch_size} (当前: {TRAINING_CONFIG['batch_size']})")
        
        # CPU优化
        cpu_count = os.cpu_count()
        recommended_workers = min(cpu_count // 2, 4)
        
        if DATA_CONFIG['num_workers'] != recommended_workers:
            logger.info(f"建议工作进程数: {recommended_workers} (当前: {DATA_CONFIG['num_workers']})")

def main():
    """主函数"""
    logger.info("🚀 车载语音识别系统 - 配置管理")
    
    config_manager = ConfigManager()
    
    # 验证配置
    if not config_manager.validate_config():
        logger.error("❌ 配置验证失败，请检查配置文件")
        return False
    
    # 打印配置摘要
    config_manager.print_config_summary()
    
    # 硬件优化建议
    config_manager.optimize_for_hardware()
    
    # 保存配置
    config_manager.save_config()
    
    logger.info("✅ 配置管理完成")
    logger.info("\n下一步: 运行 python train.py 开始训练")
    
    return True

if __name__ == "__main__":
    main()