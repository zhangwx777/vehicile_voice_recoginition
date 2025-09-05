# scripts/model_evaluation.py

import os
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib

# 设置matplotlib使用英文字体，避免中文字体缺失警告
matplotlib.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import precision_recall_curve, roc_curve, auc
import time

from settings import PATH_CONFIG, AUDIO_CONFIG, MODEL_CONFIG
from preprocessor import AudioPreprocessor
from loader import create_data_loaders
from individual_loader import create_individual_data_loaders
from cnn_model import CNNModel
from evaluator import ModelEvaluator
from recognizer import VehicleRecognizer
from logger import logger

class ComprehensiveEvaluator:
    """综合模型评估器"""
    
    def __init__(self, model_path, label_mapping_path):
        self.model_path = model_path
        self.label_mapping_path = label_mapping_path
        self.results_dir = Path(PATH_CONFIG['results_dir']) / 'evaluation'
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载模型和标签映射
        self._load_model_and_labels()
        
    def _load_model_and_labels(self):
        """加载模型和标签映射"""
        logger.info("正在加载模型和标签映射...")
        
        # 检查文件存在性
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")
        if not os.path.exists(self.label_mapping_path):
            raise FileNotFoundError(f"标签映射文件不存在: {self.label_mapping_path}")
        
        # 加载标签映射
        with open(self.label_mapping_path, 'r', encoding='utf-8') as f:
            self.label_mapping = json.load(f)
        
        self.class_names = list(self.label_mapping.keys())
        self.num_classes = len(self.class_names)
        
        logger.info(f"类别数量: {self.num_classes}")
        logger.info(f"类别名称: {self.class_names}")
        
        # 加载模型
        self.model = CNNModel(num_classes=self.num_classes, **MODEL_CONFIG)
        
        # 尝试加载模型状态
        try:
            checkpoint = torch.load(self.model_path, map_location='cpu')
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                logger.info("加载完整检查点文件")
            else:
                self.model.load_state_dict(checkpoint)
                logger.info("加载模型状态字典")
        except Exception as e:
            logger.error(f"模型加载失败: {str(e)}")
            raise
        
        self.model.eval()
        logger.info("✅ 模型加载成功")
    
    def evaluate_on_test_set(self):
        """在测试集上评估模型"""
        logger.info("🧪 在测试集上评估模型...")
        
        # 创建个体识别数据加载器
        preprocessor = AudioPreprocessor(**AUDIO_CONFIG)
        individual_data_dir = os.path.join(PATH_CONFIG['data_dir'], 'individual_vehicles')
        _, _, test_loader, _ = create_individual_data_loaders(
            individual_data_dir, preprocessor,
            batch_size=16
        )
        
        # 使用评估器
        evaluator = ModelEvaluator(self.model)
        accuracy, f1, cm, all_preds, all_labels = evaluator.evaluate_and_visualize(
            test_loader, self.class_names,
            save_path=str(self.results_dir / 'test_confusion_matrix.png')
        )
        
        # 详细分类报告
        report = classification_report(
            all_labels, all_preds, 
            target_names=self.class_names, 
            output_dict=True
        )
        
        # 保存详细报告
        self._save_classification_report(report, accuracy, f1)
        
        return {
            'accuracy': accuracy,
            'f1_score': f1,
            'confusion_matrix': cm,
            'predictions': all_preds,
            'true_labels': all_labels,
            'classification_report': report
        }
    
    def evaluate_individual_samples(self, num_samples=20):
        """评估个别样本"""
        logger.info(f"🔍 评估个别样本 (随机选择{num_samples}个)...")
        
        # 收集测试样本
        test_samples = []
        data_path = Path(PATH_CONFIG['data_dir'])
        
        for class_name in self.class_names:
            class_dir = data_path / class_name
            if class_dir.exists():
                audio_files = list(class_dir.glob("*.wav"))
                # 随机选择样本
                selected = np.random.choice(
                    audio_files, 
                    size=min(num_samples // len(self.class_names) + 1, len(audio_files)), 
                    replace=False
                )
                test_samples.extend([(f, class_name) for f in selected])
        
        # 创建识别器
        recognizer = VehicleRecognizer(self.model_path, self.label_mapping_path)
        
        # 评估每个样本
        results = []
        correct_predictions = 0
        
        for audio_file, true_label in test_samples[:num_samples]:
            try:
                start_time = time.time()
                predicted_label, confidence = recognizer.predict(str(audio_file))
                inference_time = time.time() - start_time
                
                is_correct = predicted_label == true_label
                if is_correct:
                    correct_predictions += 1
                
                results.append({
                    'file': audio_file.name,
                    'true_label': true_label,
                    'predicted_label': predicted_label,
                    'confidence': confidence,
                    'correct': is_correct,
                    'inference_time': inference_time
                })
                
                # 实时打印结果
                status = "✅" if is_correct else "❌"
                logger.info(f"{status} {audio_file.name}: {true_label} -> {predicted_label} ({confidence:.3f})")
                
            except Exception as e:
                logger.error(f"评估样本失败 {audio_file}: {str(e)}")
                continue
        
        # 计算统计信息
        sample_accuracy = correct_predictions / len(results) if results else 0
        avg_confidence = np.mean([r['confidence'] for r in results]) if results else 0
        avg_inference_time = np.mean([r['inference_time'] for r in results]) if results else 0
        
        logger.info(f"个别样本评估结果:")
        logger.info(f"  样本数量: {len(results)}")
        logger.info(f"  准确率: {sample_accuracy:.4f}")
        logger.info(f"  平均置信度: {avg_confidence:.4f}")
        logger.info(f"  平均推理时间: {avg_inference_time:.4f}秒")
        
        # 保存详细结果
        results_df = pd.DataFrame(results)
        results_df.to_csv(self.results_dir / 'individual_sample_results.csv', index=False)
        
        return {
            'results': results,
            'sample_accuracy': sample_accuracy,
            'avg_confidence': avg_confidence,
            'avg_inference_time': avg_inference_time
        }
    
    def analyze_model_performance(self):
        """分析模型性能"""
        logger.info("📊 分析模型性能...")
        
        # 加载训练历史（如果存在）
        training_history_path = self.results_dir.parent / 'training_history.png'
        if training_history_path.exists():
            logger.info(f"训练历史图表: {training_history_path}")
        
        # 模型复杂度分析
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        logger.info(f"模型复杂度分析:")
        logger.info(f"  总参数量: {total_params:,}")
        logger.info(f"  可训练参数: {trainable_params:,}")
        logger.info(f"  模型大小: {total_params * 4 / 1024 / 1024:.2f} MB")
        
        # 推理速度测试
        self._benchmark_inference_speed()
        
        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'model_size_mb': total_params * 4 / 1024 / 1024
        }
    
    def _benchmark_inference_speed(self):
        """推理速度基准测试"""
        logger.info("⏱️  进行推理速度基准测试...")
        
        # 创建虚拟输入
        duration = AUDIO_CONFIG['duration']
        sr = AUDIO_CONFIG['sample_rate']
        n_mels = AUDIO_CONFIG['n_mels']
        
        # 计算时间步数
        n_steps = int((duration * sr - AUDIO_CONFIG['n_fft']) / AUDIO_CONFIG['hop_length']) + 1
        
        dummy_input = torch.randn(1, 1, n_mels, n_steps)
        
        # 预热
        with torch.no_grad():
            for _ in range(10):
                _ = self.model(dummy_input)
        
        # 实际测试
        times = []
        with torch.no_grad():
            for _ in range(100):
                start_time = time.time()
                _ = self.model(dummy_input)
                times.append(time.time() - start_time)
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        
        logger.info(f"推理速度基准测试结果:")
        logger.info(f"  平均推理时间: {avg_time*1000:.2f} ± {std_time*1000:.2f} ms")
        logger.info(f"  理论FPS: {1/avg_time:.1f}")
    
    def _save_classification_report(self, report, accuracy, f1):
        """保存分类报告"""
        # 文本报告
        report_text = f"模型评估报告\\n{'='*50}\\n"
        report_text += f"总体准确率: {accuracy:.4f}\\n"
        report_text += f"加权F1分数: {f1:.4f}\\n\\n"
        
        report_text += "各类别详细指标:\\n"
        for class_name in self.class_names:
            if class_name in report:
                metrics = report[class_name]
                report_text += f"  {class_name}:\\n"
                report_text += f"    精确率: {metrics['precision']:.4f}\\n"
                report_text += f"    召回率: {metrics['recall']:.4f}\\n"
                report_text += f"    F1分数: {metrics['f1-score']:.4f}\\n"
                report_text += f"    样本数: {metrics['support']}\\n\\n"
        
        # 保存文本报告
        with open(self.results_dir / 'classification_report.txt', 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        # 保存JSON报告
        with open(self.results_dir / 'classification_report.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"分类报告已保存至: {self.results_dir}")
    
    def generate_evaluation_summary(self, test_results, sample_results, performance_info):
        """生成评估摘要"""
        logger.info("📋 生成评估摘要...")
        
        summary = {
            'model_info': {
                'model_path': str(self.model_path),
                'num_classes': self.num_classes,
                'class_names': self.class_names,
                'total_params': performance_info['total_params'],
                'model_size_mb': performance_info['model_size_mb']
            },
            'test_set_performance': {
                'accuracy': test_results['accuracy'],
                'f1_score': test_results['f1_score']
            },
            'sample_evaluation': {
                'sample_accuracy': sample_results['sample_accuracy'],
                'avg_confidence': sample_results['avg_confidence'],
                'avg_inference_time': sample_results['avg_inference_time']
            },
            'evaluation_timestamp': str(pd.Timestamp.now())
        }
        
        # 保存摘要
        with open(self.results_dir / 'evaluation_summary.json', 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        # 打印摘要
        logger.info("\\n🎯 评估摘要:")
        logger.info("="*60)
        logger.info(f"模型类别数: {self.num_classes}")
        logger.info(f"模型参数量: {performance_info['total_params']:,}")
        logger.info(f"模型大小: {performance_info['model_size_mb']:.2f} MB")
        logger.info(f"测试集准确率: {test_results['accuracy']:.4f}")
        logger.info(f"测试集F1分数: {test_results['f1_score']:.4f}")
        logger.info(f"样本评估准确率: {sample_results['sample_accuracy']:.4f}")
        logger.info(f"平均推理时间: {sample_results['avg_inference_time']:.4f}秒")
        logger.info("="*60)
        
        return summary

def main():
    """主函数"""
    logger.info("🎯 车载语音识别系统 - 模型评估")
    
    # 检查必要文件 - 使用个体识别模型
    model_path = os.path.join(PATH_CONFIG['checkpoints_dir'], '..', 'models', 'individual_best_model.pth')
    label_mapping_path = os.path.join(PATH_CONFIG['checkpoints_dir'], '..', 'models', 'individual_label_mapping.json')
    
    # 规范化路径
    model_path = os.path.normpath(model_path)
    label_mapping_path = os.path.normpath(label_mapping_path)
    
    if not os.path.exists(model_path):
        logger.error(f"❌ 模型文件不存在: {model_path}")
        logger.info("请先运行训练: python train.py")
        return False
    
    if not os.path.exists(label_mapping_path):
        logger.error(f"❌ 标签映射文件不存在: {label_mapping_path}")
        logger.info("请先运行训练: python train.py")
        return False
    
    try:
        # 创建评估器
        evaluator = ComprehensiveEvaluator(model_path, label_mapping_path)
        
        # 1. 测试集评估
        test_results = evaluator.evaluate_on_test_set()
        
        # 2. 个别样本评估
        sample_results = evaluator.evaluate_individual_samples(num_samples=30)
        
        # 3. 性能分析
        performance_info = evaluator.analyze_model_performance()
        
        # 4. 生成摘要
        summary = evaluator.generate_evaluation_summary(
            test_results, sample_results, performance_info
        )
        
        logger.info("✅ 模型评估完成!")
        logger.info(f"评估结果保存在: {evaluator.results_dir}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 评估过程中出错: {str(e)}")
        return False

if __name__ == "__main__":
    main()