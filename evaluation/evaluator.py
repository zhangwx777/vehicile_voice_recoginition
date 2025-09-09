# evaluation/evaluator.py

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
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from sklearn.metrics import classification_report, precision_recall_curve, roc_curve, auc
import time

from core.logger import logger
from core.settings import PATH_CONFIG, AUDIO_CONFIG, MODEL_CONFIG


class ModelEvaluator:
    """模型评估器"""
    
    def __init__(self, model, device=None):
        self.model = model
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        logger.info(f"模型评估器初始化完成，设备: {self.device}")
    
    def evaluate(self, data_loader):
        """评估模型性能"""
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs, 1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # 计算指标
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted')
        cm = confusion_matrix(all_labels, all_preds)
        
        return accuracy, f1, cm, all_preds, all_labels
    
    def evaluate_and_visualize(self, data_loader, class_names, save_path=None):
        """评估模型并生成可视化"""
        accuracy, f1, cm, all_preds, all_labels = self.evaluate(data_loader)
        
        # 生成混淆矩阵可视化
        if save_path:
            self._plot_confusion_matrix(cm, class_names, save_path)
        
        logger.info(f"评估完成 - 准确率: {accuracy:.4f}, F1分数: {f1:.4f}")
        
        return accuracy, f1, cm, all_preds, all_labels
    
    def _plot_confusion_matrix(self, cm, class_names, save_path):
        """绘制混淆矩阵"""
        try:
            plt.figure(figsize=(10, 8))
            
            # 计算百分比
            cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
            
            # 创建热力图
            sns.heatmap(cm_percent, 
                       annot=True, 
                       fmt='.1f', 
                       cmap='Blues',
                       xticklabels=class_names,
                       yticklabels=class_names,
                       cbar_kws={'label': 'Percentage (%)'})
            
            plt.title('Confusion Matrix (%)', fontsize=16, fontweight='bold')
            plt.xlabel('Predicted Label', fontsize=12)
            plt.ylabel('True Label', fontsize=12)
            plt.xticks(rotation=45)
            plt.yticks(rotation=0)
            
            # 确保保存目录存在
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"混淆矩阵已保存至: {save_path}")
            
        except Exception as e:
            logger.error(f"绘制混淆矩阵失败: {str(e)}")
    
    def get_class_metrics(self, all_labels, all_preds, class_names):
        """获取各类别的详细指标"""
        from sklearn.metrics import classification_report
        
        report = classification_report(
            all_labels, all_preds,
            target_names=class_names,
            output_dict=True
        )
        
        return report
    
    def evaluate_single_sample(self, input_tensor):
        """评估单个样本"""
        self.model.eval()
        
        with torch.no_grad():
            input_tensor = input_tensor.to(self.device)
            if len(input_tensor.shape) == 3:  # 添加batch维度
                input_tensor = input_tensor.unsqueeze(0)
            
            outputs = self.model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(outputs, dim=1)
            confidence = torch.max(probabilities, dim=1)[0]
            
            return predicted_class.cpu().item(), confidence.cpu().item(), probabilities.cpu().numpy()


class ComprehensiveEvaluator:
    """综合模型评估器（从model_evaluation.py整合）"""
    
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
        from core.model import CNNModel
        model_config = MODEL_CONFIG.copy()
        model_config['num_classes'] = self.num_classes
        self.model = CNNModel(**model_config)
        
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
        from data.preprocessor import AudioPreprocessor
        from data.loader import create_individual_data_loaders
        
        preprocessor = AudioPreprocessor(**AUDIO_CONFIG)
        individual_data_dir = os.path.join(PATH_CONFIG['data_dir'], 'individual_recognition', 'individual_vehicles')
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
        from scripts.recognizer import VehicleRecognizer
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
        report_text = f"模型评估报告\n{'='*50}\n"
        report_text += f"总体准确率: {accuracy:.4f}\n"
        report_text += f"加权F1分数: {f1:.4f}\n\n"
        
        report_text += "各类别详细指标:\n"
        for class_name in self.class_names:
            if class_name in report:
                metrics = report[class_name]
                report_text += f"  {class_name}:\n"
                report_text += f"    精确率: {metrics['precision']:.4f}\n"
                report_text += f"    召回率: {metrics['recall']:.4f}\n"
                report_text += f"    F1分数: {metrics['f1-score']:.4f}\n"
                report_text += f"    样本数: {metrics['support']}\n\n"
        
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
        logger.info("\n🎯 评估摘要:")
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