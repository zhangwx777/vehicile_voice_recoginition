# individual_loader.py

import os
import gc
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from logger import logger
from settings import DATA_CONFIG


class IndividualVehicleDataset(Dataset):
    """个体车辆音频数据集"""
    
    def __init__(self, data_dir, preprocessor, transform=None, max_samples_per_individual=None, enable_cache=False):
        self.data_dir = data_dir
        self.preprocessor = preprocessor
        self.transform = transform
        self.max_samples_per_individual = max_samples_per_individual or DATA_CONFIG.get('max_samples_per_class', 1000)
        self.feature_cache = {} if enable_cache else None

        # 收集数据和标签
        self.file_paths = []
        self.labels = []
        self.label_to_idx = {}
        self.idx_to_label = {}

        self._prepare_individual_data()
        
        # 内存优化：在数据准备后清理不必要的内存
        gc.collect()

    def _prepare_individual_data(self):
        """准备个体化数据集"""
        try:
            # 个体化数据集结构: data_dir/vehicle_type/individual_id/state/*.wav
            if not os.path.exists(self.data_dir):
                raise FileNotFoundError(f"数据目录不存在: {self.data_dir}")
                
            vehicle_types = [d for d in os.listdir(self.data_dir)
                             if os.path.isdir(os.path.join(self.data_dir, d)) and not d.endswith('.json')]
            
            if not vehicle_types:
                raise ValueError(f"在数据目录中未找到任何车辆类型文件夹: {self.data_dir}")

            # 收集所有个体车辆
            all_individuals = []
            for vehicle_type in vehicle_types:
                vehicle_type_dir = os.path.join(self.data_dir, vehicle_type)
                individuals = [d for d in os.listdir(vehicle_type_dir)
                              if os.path.isdir(os.path.join(vehicle_type_dir, d))]
                
                for individual in individuals:
                    all_individuals.append(individual)
            
            # 创建个体标签映射
            for idx, individual in enumerate(sorted(all_individuals)):  # 排序保证一致性
                self.label_to_idx[individual] = idx
                self.idx_to_label[idx] = individual

            logger.info(f"找到 {len(all_individuals)} 个个体车辆: {sorted(all_individuals)}")

            # 收集文件路径和标签
            total_files = 0
            for vehicle_type in vehicle_types:
                vehicle_type_dir = os.path.join(self.data_dir, vehicle_type)
                individuals = [d for d in os.listdir(vehicle_type_dir)
                              if os.path.isdir(os.path.join(vehicle_type_dir, d))]
                
                for individual in individuals:
                    individual_dir = os.path.join(vehicle_type_dir, individual)
                    
                    # 遍历所有状态目录（idle, acceleration, cruising, deceleration）
                    states = [d for d in os.listdir(individual_dir)
                             if os.path.isdir(os.path.join(individual_dir, d))]
                    
                    individual_files = []
                    for state in states:
                        state_dir = os.path.join(individual_dir, state)
                        
                        # 支持更多音频格式
                        audio_files = []
                        for ext in ['.wav', '.mp3', '.flac', '.ogg', '.m4a']:
                            audio_files.extend([f for f in os.listdir(state_dir) if f.lower().endswith(ext)])
                        
                        # 添加完整路径
                        for audio_file in audio_files:
                            file_path = os.path.join(state_dir, audio_file)
                            if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                                individual_files.append(file_path)
                    
                    # 限制每个个体的样本数量
                    if len(individual_files) > self.max_samples_per_individual:
                        logger.info(f"{individual}: 找到 {len(individual_files)} 个文件，限制为 {self.max_samples_per_individual} 个")
                        individual_files = individual_files[:self.max_samples_per_individual]
                    else:
                        logger.info(f"{individual}: 找到 {len(individual_files)} 个文件")
                    
                    # 添加到数据集
                    for file_path in individual_files:
                        self.file_paths.append(file_path)
                        self.labels.append(self.label_to_idx[individual])
                        total_files += 1
            
            if total_files == 0:
                raise ValueError("未找到有效的音频文件")
                
            logger.info(f"个体化数据集准备完成: 总计 {total_files} 个文件，{len(all_individuals)} 个个体")
            
        except Exception as e:
            logger.error(f"个体化数据集准备失败: {str(e)}")
            raise

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label = self.labels[idx]
        
        try:
            # 检查缓存
            if self.feature_cache is not None and file_path in self.feature_cache:
                features = self.feature_cache[file_path]
            else:
                # 预处理音频
                features = self.preprocessor.process_audio_file(file_path)
                
                # 缓存特征
                if self.feature_cache is not None:
                    self.feature_cache[file_path] = features
            
            # 确保特征是正确的形状
            if isinstance(features, dict):
                # 如果返回多个特征，使用mel频谱图
                features = features.get('mel', features.get('mfcc', list(features.values())[0]))
            
            # 转换为tensor
            if not isinstance(features, torch.Tensor):
                features = torch.FloatTensor(features)
            
            # 确保有正确的维度
            if features.dim() == 2:
                features = features.unsqueeze(0)  # 添加通道维度
            
            # 应用变换
            if self.transform:
                features = self.transform(features)
            
            return features, label
            
        except Exception as e:
            logger.error(f"处理文件失败 {file_path}: {str(e)}")
            # 返回零张量作为fallback
            dummy_features = torch.zeros((1, 128, 128))  # 默认形状
            return dummy_features, label

    def get_label_mapping(self):
        """获取标签映射"""
        return self.label_to_idx.copy()
    
    def get_class_counts(self):
        """获取每个类别的样本数量"""
        from collections import Counter
        return Counter(self.labels)
    
    def clear_cache(self):
        """清理缓存"""
        if self.feature_cache:
            self.feature_cache.clear()
            gc.collect()
            logger.info("特征缓存已清理")


def create_individual_data_loaders(data_dir, preprocessor, batch_size=None, test_size=None, val_size=None, enable_cache=False):
    """创建个体识别数据加载器"""
    
    # 使用默认配置
    batch_size = batch_size or DATA_CONFIG.get('batch_size', 32)
    test_size = test_size or DATA_CONFIG.get('test_size', 0.2)
    val_size = val_size or DATA_CONFIG.get('val_size', 0.2)
    
    logger.info(f"创建个体识别数据加载器: batch_size={batch_size}, test_size={test_size}, val_size={val_size}")
    
    try:
        # 创建完整数据集
        full_dataset = IndividualVehicleDataset(data_dir, preprocessor, enable_cache=enable_cache)
        
        if len(full_dataset) == 0:
            raise ValueError("数据集为空")
        
        # 获取文件路径和标签
        file_paths = full_dataset.file_paths
        labels = full_dataset.labels
        
        # 分层采样确保每个类别都有代表
        train_paths, temp_paths, train_labels, temp_labels = train_test_split(
            file_paths, labels, 
            test_size=test_size + val_size,
            stratify=labels,
            random_state=42
        )
        
        # 进一步分割验证集和测试集
        val_ratio = val_size / (test_size + val_size)
        val_paths, test_paths, val_labels, test_labels = train_test_split(
            temp_paths, temp_labels,
            test_size=1-val_ratio,
            stratify=temp_labels,
            random_state=42
        )
        
        logger.info(f"数据分割完成:")
        logger.info(f"  训练集: {len(train_paths)} 样本")
        logger.info(f"  验证集: {len(val_paths)} 样本")
        logger.info(f"  测试集: {len(test_paths)} 样本")
        
        # 创建子数据集
        from loader import VehicleSubDataset
        
        train_dataset = VehicleSubDataset(
            data_dir, preprocessor, train_paths, train_labels,
            full_dataset.label_to_idx, full_dataset.idx_to_label, enable_cache
        )
        
        val_dataset = VehicleSubDataset(
            data_dir, preprocessor, val_paths, val_labels,
            full_dataset.label_to_idx, full_dataset.idx_to_label, enable_cache
        )
        
        test_dataset = VehicleSubDataset(
            data_dir, preprocessor, test_paths, test_labels,
            full_dataset.label_to_idx, full_dataset.idx_to_label, enable_cache
        )
        
        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=0,  # Windows上设为0避免多进程问题
            pin_memory=torch.cuda.is_available()
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available()
        )
        
        test_loader = DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available()
        )
        
        logger.info("个体识别数据加载器创建成功")
        
        return train_loader, val_loader, test_loader, full_dataset.label_to_idx
        
    except Exception as e:
        logger.error(f"创建个体识别数据加载器失败: {str(e)}")
        raise