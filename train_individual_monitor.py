# scripts/train_individual_monitor.py

import os
import time
import psutil
import torch
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
from logger import logger

# 设置 matplotlib 使用英文字体，避免中文字体缺失警告
matplotlib.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False

class IndividualTrainingMonitor:
    """个体识别训练监控器"""
    
    def __init__(self, log_dir="logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.start_time = time.time()
        
        self.metrics = {
            'cpu_usage': [],
            'memory_usage': [],
            'gpu_usage': [],
            'gpu_memory': [],
            'timestamps': []
        }
        
    def log_system_stats(self):
        """记录系统状态"""
        current_time = time.time() - self.start_time
        
        # CPU使用率
        cpu_percent = psutil.cpu_percent()
        
        # 内存使用率
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        
        # GPU使用率（如果可用）
        gpu_usage = 0
        gpu_memory = 0
        
        if torch.cuda.is_available():
            try:
                gpu_memory_used = torch.cuda.memory_allocated(0) / 1024**3
                gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                gpu_memory = (gpu_memory_used / gpu_memory_total) * 100
                
                # 简单的GPU使用率估算
                gpu_usage = min(gpu_memory, 100)
                
            except Exception as e:
                logger.warning(f"获取GPU信息失败: {e}")
        
        # 记录指标
        self.metrics['timestamps'].append(current_time)
        self.metrics['cpu_usage'].append(cpu_percent)
        self.metrics['memory_usage'].append(memory_percent)
        self.metrics['gpu_usage'].append(gpu_usage)
        self.metrics['gpu_memory'].append(gpu_memory)
        
        # 输出当前状态
        logger.info(f"系统状态 - CPU: {cpu_percent:.1f}%, 内存: {memory_percent:.1f}%, GPU: {gpu_usage:.1f}%")
        
    def save_monitoring_report(self):
        """保存监控报告"""
        if not self.metrics['timestamps']:
            logger.warning("没有监控数据可保存")
            return
            
        try:
            # 创建图表
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('个体识别训练系统监控报告', fontsize=16)
            
            timestamps = self.metrics['timestamps']
            
            # CPU使用率
            ax1.plot(timestamps, self.metrics['cpu_usage'], 'b-', linewidth=2)
            ax1.set_title('CPU Usage (%)')
            ax1.set_xlabel('Time (seconds)')
            ax1.set_ylabel('Usage (%)')
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim(0, 100)
            
            # 内存使用率
            ax2.plot(timestamps, self.metrics['memory_usage'], 'g-', linewidth=2)
            ax2.set_title('Memory Usage (%)')
            ax2.set_xlabel('Time (seconds)')
            ax2.set_ylabel('Usage (%)')
            ax2.grid(True, alpha=0.3)
            ax2.set_ylim(0, 100)
            
            # GPU使用率
            if torch.cuda.is_available() and any(self.metrics['gpu_usage']):
                ax3.plot(timestamps, self.metrics['gpu_usage'], 'r-', linewidth=2)
                ax3.set_title('GPU Usage (%)')
            else:
                ax3.text(0.5, 0.5, 'GPU Not Available', ha='center', va='center', transform=ax3.transAxes)
                ax3.set_title('GPU Usage (N/A)')
            ax3.set_xlabel('Time (seconds)')
            ax3.set_ylabel('Usage (%)')
            ax3.grid(True, alpha=0.3)
            ax3.set_ylim(0, 100)
            
            # GPU内存使用率
            if torch.cuda.is_available() and any(self.metrics['gpu_memory']):
                ax4.plot(timestamps, self.metrics['gpu_memory'], 'm-', linewidth=2)
                ax4.set_title('GPU Memory (%)')
            else:
                ax4.text(0.5, 0.5, 'GPU Memory Not Available', ha='center', va='center', transform=ax4.transAxes)
                ax4.set_title('GPU Memory (N/A)')
            ax4.set_xlabel('Time (seconds)')
            ax4.set_ylabel('Usage (%)')
            ax4.grid(True, alpha=0.3)
            ax4.set_ylim(0, 100)
            
            plt.tight_layout()
            
            # 保存图表
            report_path = self.log_dir / 'individual_training_monitor_report.png'
            plt.savefig(report_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"监控报告已保存: {report_path}")
            
        except Exception as e:
            logger.error(f"保存监控报告失败: {e}")

def run_individual_training_with_monitoring():
    """运行带监控的个体识别训练"""
    monitor = IndividualTrainingMonitor()
    
    # 记录初始系统状态
    monitor.log_system_stats()
    
    # 执行训练
    train_success = execute_individual_training()
    
    # 记录最终系统状态
    monitor.log_system_stats()
    
    # 保存监控报告
    monitor.save_monitoring_report()
    
    return train_success

def execute_individual_training():
    """执行个体识别训练过程"""
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import json
    from helpers import set_seed, ensure_dir, count_parameters
    
    try:
        # 设置随机种子
        set_seed()
        
        # 导入必要模块时强制重新加载配置
        import importlib
        import settings
        importlib.reload(settings)  # 强制重新加载配置
        
        from settings import AUDIO_CONFIG, TRAINING_CONFIG, PATH_CONFIG, MODEL_CONFIG, DEVICE_CONFIG
        from preprocessor import AudioPreprocessor
        from individual_loader import create_individual_data_loaders
        from cnn_model import CNNModel
        from trainer import ModelTrainer
        from evaluator import ModelEvaluator
        
        # 修改路径配置为个体化数据集
        individual_data_dir = "vehicle_audio_data/individual_vehicles"
        individual_model_path = "models/individual_best_model.pth"
        individual_label_mapping_path = "models/individual_label_mapping.json"
        individual_results_dir = "results/individual"
        
        # 确保目录存在
        ensure_dir(os.path.dirname(individual_model_path))
        ensure_dir(individual_results_dir)
        ensure_dir(PATH_CONFIG['logs_dir'])
        ensure_dir(PATH_CONFIG['checkpoints_dir'])
        
        logger.info("机动车个体声纹识别系统 - 训练模块")
        logger.info(f"使用设备: {DEVICE_CONFIG['device']}")
        logger.info(f"个体化数据目录: {individual_data_dir}")
        logger.info(f"早停配置: patience={TRAINING_CONFIG['early_stopping']['patience']}, min_delta={TRAINING_CONFIG['early_stopping']['min_delta']}")
        
        # 检查个体化数据集是否存在
        if not os.path.exists(individual_data_dir):
            logger.error(f"个体化数据集不存在: {individual_data_dir}")
            logger.info("请先运行: python generate_individual_dataset.py")
            return False
        
        # 初始化预处理器（启用数据增强）
        preprocessor = AudioPreprocessor(
            sample_rate=AUDIO_CONFIG['sample_rate'],
            n_fft=AUDIO_CONFIG['n_fft'],
            hop_length=AUDIO_CONFIG['hop_length'],
            n_mels=AUDIO_CONFIG['n_mels'],
            duration=AUDIO_CONFIG['duration'],
            enable_augmentation=True,
            augmentation_config=AUDIO_CONFIG.get('augmentation_config', {}),
            extract_features=AUDIO_CONFIG.get('extract_features', ['mel'])
        )
        
        # 创建数据加载器（使用个体化数据集）
        logger.info("正在创建个体识别数据加载器...")
        train_loader, val_loader, test_loader, label_mapping = create_individual_data_loaders(
            individual_data_dir, preprocessor,
            batch_size=TRAINING_CONFIG['batch_size'],
            enable_cache=False  # 训练时不启用缓存以节约内存
        )
        
        logger.info(f"训练集: {len(train_loader.dataset)} 样本")
        logger.info(f"验证集: {len(val_loader.dataset)} 样本")
        logger.info(f"测试集: {len(test_loader.dataset)} 样本")
        logger.info(f"个体数量: {len(label_mapping)}")
        logger.info(f"个体映射: {label_mapping}")
        
        # 创建模型（调整类别数为个体数量）
        model = CNNModel(
            num_classes=len(label_mapping),
            input_channels=MODEL_CONFIG.get('input_channels', 1),
            base_filters=MODEL_CONFIG.get('base_filters', 32),
            num_conv_blocks=MODEL_CONFIG.get('num_conv_blocks', 4),
            dropout_rate=MODEL_CONFIG.get('dropout_rate', 0.5)
        )
        logger.info(f"模型参数量: {count_parameters(model):,}")
        
        # 定义损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(),
                               lr=TRAINING_CONFIG['learning_rate'],
                               weight_decay=TRAINING_CONFIG['weight_decay'])
        
        # 学习率调度器
        scheduler = optim.lr_scheduler.StepLR(optimizer,
                                              step_size=TRAINING_CONFIG['step_size'],
                                              gamma=TRAINING_CONFIG['gamma'])
        
        # 训练模型
        logger.info("开始个体识别训练...")
        trainer = ModelTrainer(model)
        
        best_val_acc = trainer.train(
            train_loader, val_loader, criterion, optimizer, scheduler,
            num_epochs=TRAINING_CONFIG['num_epochs'],
            model_save_path=individual_model_path,
            enable_early_stopping=True,
            save_checkpoints=True
        )
        
        # 评估模型
        logger.info("开始个体识别评估...")
        evaluator = ModelEvaluator(model)
        accuracy, f1, cm, all_preds, all_labels = evaluator.evaluate_and_visualize(
            test_loader, list(label_mapping.keys()),
            save_path=os.path.join(individual_results_dir, 'individual_confusion_matrix.png')
        )
        
        logger.info("\n个体识别测试集性能:")
        logger.info(f"准确率: {accuracy:.4f}")
        logger.info(f"F1分数: {f1:.4f}")
        logger.info(f"最佳验证准确率: {best_val_acc:.4f}")
        
        # 保存个体标签映射
        ensure_dir(os.path.dirname(individual_label_mapping_path))
        with open(individual_label_mapping_path, 'w', encoding='utf-8') as f:
            json.dump(label_mapping, f, ensure_ascii=False, indent=2)
        
        logger.info("\n个体识别训练和评估完成!")
        logger.info(f"模型已保存至: {individual_model_path}")
        logger.info(f"标签映射已保存至: {individual_label_mapping_path}")
        logger.info(f"结果可视化已保存至: {individual_results_dir}")
        
        # 自动生成可视化结果
        logger.info("\n🎨 开始生成个体识别可视化结果...")
        try:
            from generate_visualizations import generate_audio_visualizations, generate_prediction_visualizations
            
            # 生成音频特征可视化
            generate_audio_visualizations()
            
            # 生成预测结果可视化
            generate_prediction_visualizations()
            
            logger.info("✅ 个体识别可视化结果生成完成!")
            logger.info("📁 可视化文件位置:")
            logger.info("   - results/audio_visualizations/")
            logger.info("   - results/prediction_visualizations/")
            
        except Exception as e:
            logger.warning(f"⚠️  可视化生成失败: {str(e)}")
            logger.info("💡 您可以稍后手动运行: python generate_visualizations.py")
        
        return True
        
    except Exception as e:
        logger.error(f"个体识别训练执行失败: {str(e)}")
        import traceback
        logger.error(f"详细错误: {traceback.format_exc()}")
        return False

def main():
    """主函数"""
    logger.info("🎯 车载语音个体识别系统 - 训练执行器")
    
    # 检查前置条件
    individual_data_dir = "vehicle_audio_data/individual_vehicles"
    
    # 检查个体化数据是否存在
    if not os.path.exists(individual_data_dir):
        logger.error(f"❌ 个体化数据目录不存在: {individual_data_dir}")
        logger.info("请先运行: python generate_individual_dataset.py")
        return False
    
    # 检查数据文件
    data_files = []
    for root, dirs, files in os.walk(individual_data_dir):
        data_files.extend([f for f in files if f.endswith(('.wav', '.mp3'))])
    
    if len(data_files) < 10:
        logger.error(f"❌ 个体化数据文件过少: {len(data_files)} 个文件")
        logger.info("请先运行: python generate_individual_dataset.py")
        return False
    
    logger.info(f"✅ 发现 {len(data_files)} 个个体化音频文件")
    
    # 开始个体识别训练
    success = run_individual_training_with_monitoring()
    
    if success:
        logger.info("🎉 个体识别训练流程完成!")
        logger.info("下一步: 运行 python vehicle_inference.py <audio_file> 进行个体识别测试")
    else:
        logger.error("💥 个体识别训练流程失败!")
    
    return success

if __name__ == "__main__":
    main()