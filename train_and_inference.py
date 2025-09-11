#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
车辆声纹识别系统完整训练和推理脚本
包含数据加载、模型训练、评估和推理功能
"""

import os
import sys
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import librosa
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 设置英文字体和样式
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
sns.set_palette("husl")
from typing import Dict, List, Tuple, Optional, Any
from tqdm import tqdm

# 导入项目配置和模块
from config import *
from core.logger import logger
from data.preprocessor import AudioPreprocessor
from enhanced_feature_extractor import EnhancedAudioFeatureExtractor
from vector_similarity_engine import VectorSimilarityEngine, VehicleProfile
# 训练可视化功能已集成到主脚本中

class SimpleCNNModel(nn.Module):
    """简化的CNN模型用于车辆声纹识别"""
    
    def __init__(self, input_shape, num_classes):
        super(SimpleCNNModel, self).__init__()
        
        # 计算输入特征数
        if len(input_shape) == 2:
            # 梅尔频谱图输入 (n_mels, time_frames)
            self.input_channels = 1
            self.input_height = input_shape[0]
            self.input_width = input_shape[1]
        else:
            raise ValueError(f"不支持的输入形状: {input_shape}")
        
        # 卷积层
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # 池化层
        self.pool = nn.MaxPool2d(2, 2)
        
        # Dropout
        self.dropout = nn.Dropout(0.3)
        
        # 计算全连接层输入大小
        self._calculate_fc_input_size()
        
        # 全连接层
        self.fc1 = nn.Linear(self.fc_input_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
        
        # 激活函数
        self.relu = nn.ReLU()
        
    def _calculate_fc_input_size(self):
        """计算全连接层输入大小"""
        # 模拟前向传播计算特征图大小
        x = torch.randn(1, 1, self.input_height, self.input_width)
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        self.fc_input_size = x.view(1, -1).size(1)
        
    def forward(self, x):
        # 卷积层
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        
        # 展平
        x = x.view(x.size(0), -1)
        
        # 全连接层
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.fc3(x)
        
        return x

class VehicleAudioDataset(Dataset):
    """车辆音频数据集"""
    
    def __init__(self, audio_paths, labels, preprocessor):
        self.audio_paths = audio_paths
        self.labels = labels
        self.preprocessor = preprocessor
        
    def __len__(self):
        return len(self.audio_paths)
    
    def __getitem__(self, idx):
        audio_path = self.audio_paths[idx]
        label = self.labels[idx]
        
        # 预处理音频
        features = self.preprocessor.preprocess_audio(audio_path)
        if features is None:
            # 如果预处理失败，返回零特征
            features = np.zeros((128, 94))  # 默认梅尔频谱图大小
        
        # 转换为tensor
        features = torch.FloatTensor(features).unsqueeze(0)  # 添加通道维度
        label = torch.LongTensor([label])[0]
        
        return features, label

class VehicleRecognitionTrainer:
    """车辆声纹识别训练器"""
    
    def __init__(self, data_dir: str = None, enable_visualization: bool = True):
        """
        初始化训练器
        
        Args:
            data_dir: 数据目录路径
            enable_visualization: 是否启用训练可视化
        """
        self.data_dir = data_dir or PATH_CONFIG['individual_data_dir']
        self.models_dir = Path(PATH_CONFIG['models_dir'])
        self.models_dir.mkdir(exist_ok=True)
        
        # 初始化组件
        self.preprocessor = AudioPreprocessor()
        self.enhanced_extractor = None
        self.similarity_engine = None
        
        # 训练相关
        self.model = None
        self.label_encoder = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 可视化
        self.enable_visualization = enable_visualization
        self.training_history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': [], 'learning_rate': []}
        
        # 训练配置初始化完成
        
    def load_dataset(self) -> Tuple[List[str], List[str], List[str]]:
        """
        加载数据集
        
        Returns:
            Tuple[List[str], List[str], List[str]]: 音频路径、车辆ID、车辆类型
        """
        print("加载数据集...")
        
        audio_paths = []
        vehicle_ids = []
        vehicle_types = []
        
        data_path = Path(self.data_dir)
        if not data_path.exists():
            raise FileNotFoundError(f"数据目录不存在: {self.data_dir}")
        
        # 检查实际数据路径
        individual_vehicles_path = data_path / 'individual_vehicles'
        if individual_vehicles_path.exists():
            data_path = individual_vehicles_path
        
        # 遍历车辆目录
        for vehicle_dir in data_path.iterdir():
            if not vehicle_dir.is_dir():
                continue
                
            vehicle_id = vehicle_dir.name
            vehicle_type = vehicle_id.split('_')[0]  # 提取车辆类型
            
            # 遍历音频文件
            for audio_file in vehicle_dir.glob('*.wav'):
                audio_paths.append(str(audio_file))
                vehicle_ids.append(vehicle_id)
                vehicle_types.append(vehicle_type)
        
        print(f"加载完成: {len(audio_paths)} 个音频文件，{len(set(vehicle_ids))} 个车辆个体")
        
        return audio_paths, vehicle_ids, vehicle_types
    
    def train_classification_model(self, epochs: int = 50, batch_size: int = 8) -> Dict[str, Any]:
        """
        训练分类模型
        
        Args:
            epochs: 训练轮数
            batch_size: 批次大小
            
        Returns:
            Dict[str, Any]: 训练历史
        """
        logger.info("开始训练分类模型...")
        
        # 加载数据
        audio_paths, vehicle_ids, vehicle_types = self.load_dataset()
        
        # 编码标签
        self.label_encoder = LabelEncoder()
        encoded_labels = self.label_encoder.fit_transform(vehicle_ids)
        
        # 划分数据集
        X_train, X_test, y_train, y_test = train_test_split(
            audio_paths, encoded_labels, test_size=0.2, random_state=42, stratify=encoded_labels
        )
        
        # 创建数据集
        train_dataset = VehicleAudioDataset(X_train, y_train, self.preprocessor)
        test_dataset = VehicleAudioDataset(X_test, y_test, self.preprocessor)
        
        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # 获取输入形状
        sample_features, _ = train_dataset[0]
        input_shape = sample_features.shape[1:]  # 去掉batch维度
        num_classes = len(self.label_encoder.classes_)
        
        logger.info(f"输入形状: {input_shape}")
        logger.info(f"类别数量: {num_classes}")
        
        # 创建模型
        self.model = SimpleCNNModel(input_shape, num_classes).to(self.device)
        
        # 定义损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
        
        # 训练历史
        history = {
            'train_loss': [],
            'train_acc': [],
            'test_loss': [],
            'test_acc': []
        }
        
        best_acc = 0.0
        
        # 初始化可视化
        if self.enable_visualization:
            logger.info("训练可视化已启用")
        
        # 训练循环
        for epoch in range(epochs):
            # 训练阶段
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]')
            for batch_idx, (data, target) in enumerate(train_pbar):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                train_total += target.size(0)
                train_correct += (predicted == target).sum().item()
                
                # 更新进度条
                train_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100.*train_correct/train_total:.2f}%'
                })
            
            # 测试阶段
            self.model.eval()
            test_loss = 0.0
            test_correct = 0
            test_total = 0
            
            with torch.no_grad():
                test_pbar = tqdm(test_loader, desc=f'Epoch {epoch+1}/{epochs} [Test]')
                for data, target in test_pbar:
                    data, target = data.to(self.device), target.to(self.device)
                    output = self.model(data)
                    loss = criterion(output, target)
                    
                    test_loss += loss.item()
                    _, predicted = torch.max(output.data, 1)
                    test_total += target.size(0)
                    test_correct += (predicted == target).sum().item()
                    
                    test_pbar.set_postfix({
                        'Loss': f'{loss.item():.4f}',
                        'Acc': f'{100.*test_correct/test_total:.2f}%'
                    })
            
            # 计算平均损失和准确率
            train_loss_avg = train_loss / len(train_loader)
            train_acc = 100. * train_correct / train_total
            test_loss_avg = test_loss / len(test_loader)
            test_acc = 100. * test_correct / test_total
            
            # 记录历史
            history['train_loss'].append(train_loss_avg)
            history['train_acc'].append(train_acc)
            history['test_loss'].append(test_loss_avg)
            history['test_acc'].append(test_acc)
            
            # 更新学习率
            scheduler.step()
            
            # 保存最佳模型
            if test_acc > best_acc:
                best_acc = test_acc
                self.save_model()
            
            logger.info(f'Epoch {epoch+1}/{epochs}: '
                       f'Train Loss: {train_loss_avg:.4f}, Train Acc: {train_acc:.2f}%, '
                       f'Test Loss: {test_loss_avg:.4f}, Test Acc: {test_acc:.2f}%')
            
            # 更新训练历史
            if self.enable_visualization:
                self.training_history['train_loss'].append(train_loss_avg)
                self.training_history['train_acc'].append(train_acc/100.0)
                self.training_history['test_loss'].append(test_loss_avg)
                self.training_history['test_acc'].append(test_acc/100.0)
                self.training_history['learning_rate'].append(scheduler.get_last_lr()[0])
        
        logger.info(f"训练完成! 最佳测试准确率: {best_acc:.2f}%")
        
        # 保存训练历史
        if self.enable_visualization:
            viz_path = Path(PATH_CONFIG['models_dir']) / 'training_history.json'
            with open(viz_path, 'w', encoding='utf-8') as f:
                json.dump(self.training_history, f, ensure_ascii=False, indent=2)
            logger.info(f"训练历史已保存到: {viz_path}")
        
        # 生成详细评估报告
        self.evaluate_model(test_loader)
        
        return history
    
    def evaluate_model(self, test_loader):
        """
        评估模型性能
        
        Args:
            test_loader: 测试数据加载器
        """
        logger.info("评估模型性能...")
        
        self.model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in tqdm(test_loader, desc='Evaluating'):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                _, predicted = torch.max(output, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        # 生成分类报告
        class_names = self.label_encoder.classes_
        report = classification_report(all_targets, all_predictions, 
                                     target_names=class_names, output_dict=True)
        
        logger.info("分类报告:")
        logger.info(classification_report(all_targets, all_predictions, target_names=class_names))
        
        # 保存评估结果
        eval_results = {
            'classification_report': report,
            'confusion_matrix': confusion_matrix(all_targets, all_predictions).tolist(),
            'class_names': class_names.tolist()
        }
        
        eval_path = self.models_dir / 'evaluation_results.json'
        with open(eval_path, 'w', encoding='utf-8') as f:
            json.dump(eval_results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"评估结果已保存到: {eval_path}")
    
    def save_model(self):
        """
        保存模型和标签编码器
        """
        # 保存模型
        model_path = self.models_dir / 'individual_best_model.pth'
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'input_shape': self.model.input_height,
            'num_classes': len(self.label_encoder.classes_)
        }, model_path)
        
        # 保存标签映射
        label_mapping = {label: idx for idx, label in enumerate(self.label_encoder.classes_)}
        label_path = self.models_dir / 'individual_label_mapping.json'
        with open(label_path, 'w', encoding='utf-8') as f:
            json.dump(label_mapping, f, ensure_ascii=False, indent=2)
        
        logger.info(f"模型已保存到: {model_path}")
        logger.info(f"标签映射已保存到: {label_path}")
    
    def build_similarity_database(self):
        """
        构建相似度搜索数据库
        """
        logger.info("构建相似度搜索数据库...")
        
        try:
            # 初始化增强特征提取器
            self.enhanced_extractor = EnhancedAudioFeatureExtractor()
            self.similarity_engine = VectorSimilarityEngine(
                feature_extractor=self.enhanced_extractor
            )
            
            # 加载数据
            audio_paths, vehicle_ids, vehicle_types = self.load_dataset()
            
            # 为每个车辆添加档案
            for audio_path, vehicle_id, vehicle_type in tqdm(
                zip(audio_paths, vehicle_ids, vehicle_types), 
                total=len(audio_paths), 
                desc="添加车辆档案"
            ):
                metadata = {
                    'vehicle_type': vehicle_type,
                    'audio_file': os.path.basename(audio_path)
                }
                
                success = self.similarity_engine.add_vehicle_profile(
                    vehicle_id, audio_path, metadata
                )
                
                if not success:
                    logger.warning(f"添加车辆档案失败: {vehicle_id}")
            
            # 保存数据库
            db_path = self.models_dir / 'similarity_database.json'
            self.similarity_engine.save_database(str(db_path))
            
            logger.info(f"相似度数据库已保存到: {db_path}")
            logger.info(f"总共添加了 {len(self.similarity_engine.vehicle_profiles)} 个车辆档案")
            
        except Exception as e:
            logger.error(f"构建相似度数据库失败: {e}")
    
    def inference_example(self, test_audio_path: str):
        """
        推理示例
        
        Args:
            test_audio_path: 测试音频路径
        """
        logger.info(f"推理示例: {test_audio_path}")
        
        if not os.path.exists(test_audio_path):
            logger.error(f"测试音频文件不存在: {test_audio_path}")
            return
        
        # 1. 分类模式推理
        if self.model is not None and self.label_encoder is not None:
            logger.info("=== 分类模式推理 ===")
            
            # 预处理音频
            features = self.preprocessor.preprocess_audio(test_audio_path)
            if features is not None:
                # 转换为tensor
                input_tensor = torch.FloatTensor(features).unsqueeze(0).unsqueeze(0).to(self.device)
                
                # 模型推理
                self.model.eval()
                with torch.no_grad():
                    output = self.model(input_tensor)
                    probabilities = torch.softmax(output, dim=1)
                    confidence, predicted_idx = torch.max(probabilities, 1)
                
                # 获取预测结果
                predicted_vehicle = self.label_encoder.classes_[predicted_idx.item()]
                confidence_score = confidence.item()
                
                logger.info(f"预测车辆: {predicted_vehicle}")
                logger.info(f"置信度: {confidence_score:.4f}")
                
                # 显示前5个最可能的类别
                top5_probs, top5_indices = torch.topk(probabilities[0], 5)
                logger.info("前5个最可能的车辆:")
                for i, (prob, idx) in enumerate(zip(top5_probs, top5_indices)):
                    vehicle_name = self.label_encoder.classes_[idx.item()]
                    logger.info(f"  {i+1}. {vehicle_name}: {prob.item():.4f}")
        
        # 2. 相似度搜索推理
        if self.similarity_engine is not None:
            logger.info("\n=== 相似度搜索推理 ===")
            
            similar_vehicles = self.similarity_engine.search_similar_vehicles(
                test_audio_path, top_k=5
            )
            
            if similar_vehicles:
                logger.info("最相似的车辆:")
                for i, (vehicle_id, similarity_score, metadata) in enumerate(similar_vehicles):
                    logger.info(f"  {i+1}. {vehicle_id}: {similarity_score:.4f} ({metadata.get('vehicle_type', 'unknown')})")
            else:
                logger.info("未找到相似的车辆")

def main():
    """主函数 - 自动执行完整训练流程"""
    print("=== 车辆声纹识别系统训练流程 ===")
    
    # 初始化训练器
    trainer = VehicleRecognitionTrainer()
    
    try:
        # 步骤1: 训练分类模型
        print("训练分类模型...")
        start_time = time.time()
        history = trainer.train_classification_model(epochs=30, batch_size=8)
        print(f"分类模型训练完成，耗时: {time.time() - start_time:.2f} 秒")
        
        # 步骤2: 构建相似度数据库
        print("构建相似度数据库...")
        start_time = time.time()
        trainer.build_similarity_database()
        print(f"相似度数据库构建完成，耗时: {time.time() - start_time:.2f} 秒")
        
        # 步骤3: 推理测试
        print("推理测试...")
        test_audio_path = "vehicle_audio_data/individual_recognition/individual_vehicles/sedan_000/sedan_000_sample_000.wav"
        if os.path.exists(test_audio_path):
            trainer.inference_example(test_audio_path)
        else:
            print(f"测试音频文件不存在: {test_audio_path}")
        
        print("训练流程完成!")
        
    except Exception as e:
        print(f"训练流程出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()