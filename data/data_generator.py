# data/data_generator.py

import os
import numpy as np
import soundfile as sf
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from core.logger import logger
from core.settings import PATH_CONFIG, AUDIO_CONFIG


# 基础车辆类型生成代码已删除，只保留个体识别功能


class IndividualDataGenerator:
    """个体识别数据生成器"""
    
    def __init__(self):
        self.sample_rate = AUDIO_CONFIG.get('sample_rate', 16000)
        self.duration = AUDIO_CONFIG.get('duration', 3.0)
        self.vehicle_types = ['sedan', 'suv', 'truck', 'motorcycle', 'bus']
        
        # 个体特征参数
        self.individual_params = {
            'engine_signature': {},  # 发动机特征
            'exhaust_pattern': {},   # 排气模式
            'mechanical_noise': {},  # 机械噪声
        }
    
    def _get_vehicle_base_params(self, vehicle_type: str) -> Dict:
        """获取车辆类型的基础参数"""
        base_params = {
            'sedan': {
                'base_freq_range': (90, 130),
                'harmonics': [1, 2, 3, 4],
                'harmonic_weights': [1.0, 0.5, 0.3, 0.2],
                'noise_level': 0.02,
                'modulation_depth': 0.1
            },
            'suv': {
                'base_freq_range': (70, 110),
                'harmonics': [1, 2, 3, 4, 5],
                'harmonic_weights': [1.0, 0.6, 0.4, 0.25, 0.15],
                'noise_level': 0.025,
                'modulation_depth': 0.12
            },
            'truck': {
                'base_freq_range': (40, 80),
                'harmonics': [1, 2, 3, 4, 6, 8],
                'harmonic_weights': [1.0, 0.7, 0.5, 0.3, 0.2, 0.1],
                'noise_level': 0.04,
                'modulation_depth': 0.15
            },
            'motorcycle': {
                'base_freq_range': (180, 250),
                'harmonics': [1, 2, 3, 5, 7, 9],
                'harmonic_weights': [1.0, 0.6, 0.8, 0.4, 0.3, 0.2],
                'noise_level': 0.03,
                'modulation_depth': 0.2
            },
            'bus': {
                'base_freq_range': (50, 90),
                'harmonics': [1, 2, 3, 4, 5],
                'harmonic_weights': [1.0, 0.5, 0.3, 0.2, 0.15],
                'noise_level': 0.035,
                'modulation_depth': 0.13
            }
        }
        return base_params.get(vehicle_type, base_params['sedan'])
    
    def _generate_individual_signature(self, vehicle_type: str, individual_id: str) -> Dict:
        """为特定个体生成独特的声音特征"""
        # 使用个体ID作为随机种子，确保一致性
        np.random.seed(hash(individual_id) % 2**32)
        
        base_params = self._get_vehicle_base_params(vehicle_type)
        
        # 个体化参数
        signature = {
            'base_freq': np.random.uniform(*base_params['base_freq_range']),
            'harmonic_shift': np.random.uniform(-0.1, 0.1),  # 谐波偏移
            'amplitude_variation': np.random.uniform(0.8, 1.2),  # 幅度变化
            'phase_offset': np.random.uniform(0, 2*np.pi),  # 相位偏移
            'modulation_freq': np.random.uniform(1, 5),  # 调制频率
            'noise_color': np.random.uniform(0.5, 2.0),  # 噪声颜色
            'envelope_shape': np.random.uniform(0.2, 0.8),  # 包络形状
        }
        
        # 重置随机种子
        np.random.seed()
        
        return signature
    
    def generate_individual_audio(self, vehicle_type: str, individual_id: str, 
                                sample_idx: int = 0) -> np.ndarray:
        """为特定个体生成音频"""
        # 获取个体特征
        if individual_id not in self.individual_params['engine_signature']:
            self.individual_params['engine_signature'][individual_id] = \
                self._generate_individual_signature(vehicle_type, individual_id)
        
        signature = self.individual_params['engine_signature'][individual_id]
        base_params = self._get_vehicle_base_params(vehicle_type)
        
        # 生成时间轴
        t = np.linspace(0, self.duration, int(self.sample_rate * self.duration))
        
        # 生成基础信号
        signal = np.zeros_like(t)
        
        for i, (harmonic, weight) in enumerate(zip(base_params['harmonics'], 
                                                  base_params['harmonic_weights'])):
            freq = signature['base_freq'] * harmonic * (1 + signature['harmonic_shift'])
            phase = signature['phase_offset'] + i * np.pi / 4
            
            # 添加调制
            modulation = 1 + base_params['modulation_depth'] * \
                        np.sin(2 * np.pi * signature['modulation_freq'] * t)
            
            harmonic_signal = weight * signature['amplitude_variation'] * \
                            np.sin(2 * np.pi * freq * t + phase) * modulation
            
            signal += harmonic_signal
        
        # 添加包络
        envelope_decay = signature['envelope_shape']
        envelope = np.exp(-t * envelope_decay) * (1 + 0.2 * np.sin(2 * np.pi * 3 * t))
        signal *= envelope
        
        # 添加个体化噪声
        noise_power = base_params['noise_level'] * signature['noise_color']
        noise = np.random.normal(0, noise_power, len(t))
        
        # 添加一些随机变化（模拟驾驶条件变化）
        variation_factor = 1 + 0.1 * np.sin(2 * np.pi * 0.5 * t + sample_idx)
        signal = signal * variation_factor + noise
        
        # 归一化
        if np.max(np.abs(signal)) > 0:
            signal = signal / np.max(np.abs(signal)) * 0.8
        
        return signal.astype(np.float32)
    
    def create_vehicle_registry(self, num_individuals_per_type: int = 10) -> Dict[str, List[str]]:
        """创建车辆注册表"""
        registry = {}
        
        for vehicle_type in self.vehicle_types:
            individuals = []
            for i in range(num_individuals_per_type):
                # 生成个体ID
                individual_id = f"{vehicle_type}_{i:03d}"
                individuals.append(individual_id)
            
            registry[vehicle_type] = individuals
        
        return registry
    
    def generate_individual_dataset(self, output_dir: str, 
                                  num_individuals_per_type: int = 10,
                                  samples_per_individual: int = 20) -> bool:
        """生成个体识别数据集"""
        try:
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"开始生成个体识别数据集到: {output_dir}")
            
            # 创建车辆注册表
            registry = self.create_vehicle_registry(num_individuals_per_type)
            
            # 保存注册表
            registry_path = os.path.join(output_dir, 'vehicle_registry.json')
            with open(registry_path, 'w', encoding='utf-8') as f:
                json.dump(registry, f, indent=2, ensure_ascii=False)
            
            # 创建标签映射
            label_mapping = {}
            label_idx = 0
            
            for vehicle_type, individuals in registry.items():
                for individual_id in individuals:
                    label_mapping[individual_id] = label_idx
                    label_idx += 1
            
            # 保存标签映射
            mapping_path = os.path.join(output_dir, 'individual_label_mapping.json')
            with open(mapping_path, 'w', encoding='utf-8') as f:
                json.dump(label_mapping, f, indent=2, ensure_ascii=False)
            
            total_generated = 0
            
            # 生成每个个体的音频数据
            for vehicle_type, individuals in registry.items():
                logger.info(f"生成 {vehicle_type} 类型的个体音频...")
                
                for individual_id in individuals:
                    # 创建个体目录
                    individual_dir = os.path.join(output_dir, 'individual_vehicles', individual_id)
                    os.makedirs(individual_dir, exist_ok=True)
                    
                    # 生成该个体的多个样本
                    for sample_idx in range(samples_per_individual):
                        # 生成音频
                        audio = self.generate_individual_audio(vehicle_type, individual_id, sample_idx)
                        
                        # 保存文件
                        filename = f"{individual_id}_sample_{sample_idx:03d}.wav"
                        filepath = os.path.join(individual_dir, filename)
                        sf.write(filepath, audio, self.sample_rate)
                        
                        total_generated += 1
                    
                    if len(individuals) <= 20 or individuals.index(individual_id) % 5 == 0:
                        logger.info(f"  已完成个体 {individual_id}")
            
            logger.info(f"✅ 个体识别数据集生成完成")
            logger.info(f"   - 总个体数: {len(label_mapping)}")
            logger.info(f"   - 总样本数: {total_generated}")
            logger.info(f"   - 注册表: {registry_path}")
            logger.info(f"   - 标签映射: {mapping_path}")
            
            # 内存清理
            import gc
            gc.collect()
            logger.info("数据集生成完成，已清理内存")
            
            return True
            
        except Exception as e:
            logger.error(f"生成个体识别数据集失败: {e}")
            return False


def generate_individual_dataset_only(output_dir: str = None, 
                                   individual_count: int = 10, individual_samples: int = 20):
    """只生成个体识别数据集"""
    try:
        if output_dir is None:
            output_dir = PATH_CONFIG.get('data_dir', 'vehicle_audio_data')
        
        logger.info("🚀 开始生成个体识别数据集")
        
        # 生成个体识别数据集
        individual_generator = IndividualDataGenerator()
        individual_dir = os.path.join(output_dir, 'individual_recognition')
        if not individual_generator.generate_individual_dataset(
            individual_dir, individual_count, individual_samples):
            return False
        
        logger.info("✅ 个体识别数据集生成完成")
        return True
        
    except (FileNotFoundError, PermissionError, OSError) as e:
        logger.error(f"文件系统错误: {e}")
        return False
    except (ValueError, TypeError) as e:
        logger.error(f"参数错误: {e}")
        return False
    except Exception as e:
        logger.error(f"生成个体识别数据集失败: {e}")
        return False


if __name__ == "__main__":
    # 生成个体识别数据集
    success = generate_individual_dataset_only(
        individual_count=5,
        individual_samples=10
    )
    
    if success:
        logger.info("个体识别数据集生成成功！")
    else:
        logger.error("个体识别数据集生成失败！")