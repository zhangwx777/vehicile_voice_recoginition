# evaluation/evaluator.py

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from pathlib import Path
from logger import logger
from settings import PATH_CONFIG


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