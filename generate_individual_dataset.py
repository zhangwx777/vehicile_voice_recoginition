# generate_individual_dataset.py
# 车辆个体识别数据集生成器

import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import soundfile as sf

from logger import logger
from settings import PATH_CONFIG
from generate_individual_data import VehicleIndividualGenerator
from individual_label_manager import IndividualLabelManager

class IndividualDatasetGenerator:
    """车辆个体识别数据集生成器"""
    
    def __init__(self, data_dir: str = None, sample_rate: int = 16000):
        self.data_dir = Path(data_dir or PATH_CONFIG['data_dir']) / "individual_vehicles"
        self.sample_rate = sample_rate
        
        # 初始化组件
        self.vehicle_generator = VehicleIndividualGenerator(sample_rate=sample_rate)
        self.label_manager = IndividualLabelManager(data_dir=str(self.data_dir.parent))
        
        # 数据生成配置
        self.states = ['idle', 'acceleration', 'cruising', 'deceleration']
        self.samples_per_state = 10  # 每个状态生成的样本数
        self.audio_duration = 3.0    # 音频时长（秒）
        
        # 统计信息
        self.generation_stats = {
            'total_vehicles': 0,
            'total_samples': 0,
            'samples_by_type': {},
            'samples_by_state': {},
            'generation_time': 0
        }
    
    def create_vehicle_registry(self, vehicles_config: Dict) -> bool:
        """创建车辆注册表"""
        logger.info("📝 创建车辆注册表...")
        
        try:
            # 确保数据目录存在
            self.data_dir.mkdir(parents=True, exist_ok=True)
            
            # 保存车辆注册表
            registry_file = self.data_dir / "vehicle_registry.json"
            with open(registry_file, 'w', encoding='utf-8') as f:
                json.dump(vehicles_config, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✅ 车辆注册表已创建: {registry_file}")
            logger.info(f"   注册车辆数: {len(vehicles_config)}")
            
            # 统计各类型车辆数量
            type_counts = {}
            for vehicle_id, info in vehicles_config.items():
                vehicle_type = info['type']
                type_counts[vehicle_type] = type_counts.get(vehicle_type, 0) + 1
            
            logger.info("   车辆类型分布:")
            for vehicle_type, count in type_counts.items():
                logger.info(f"     {vehicle_type}: {count} 辆")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 创建车辆注册表失败: {str(e)}")
            return False
    
    def generate_vehicle_audio(self, vehicle_id: str, vehicle_info: Dict, 
                              state: str, sample_idx: int) -> Tuple[np.ndarray, str]:
        """为指定车辆和状态生成音频"""
        try:
            # 生成个体化音频
            audio_data = self.vehicle_generator.generate_individual_audio(
                vehicle_id=vehicle_id,
                vehicle_type=vehicle_info['type'],
                state=state,
                duration=self.audio_duration,
                variation_seed=sample_idx  # 使用样本索引作为变化种子
            )
            
            # 生成文件名
            filename = f"{vehicle_id}_{state}_{sample_idx:03d}.wav"
            
            return audio_data, filename
            
        except Exception as e:
            logger.error(f"❌ 生成车辆音频失败 {vehicle_id}-{state}-{sample_idx}: {str(e)}")
            return None, None
    
    def save_audio_file(self, audio_data: np.ndarray, file_path: Path) -> bool:
        """保存音频文件"""
        try:
            # 确保目录存在
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 保存音频文件
            sf.write(str(file_path), audio_data, self.sample_rate)
            return True
            
        except Exception as e:
            logger.error(f"❌ 保存音频文件失败 {file_path}: {str(e)}")
            return False
    
    def generate_vehicle_dataset(self, vehicle_id: str, vehicle_info: Dict) -> bool:
        """为单个车辆生成完整数据集"""
        vehicle_type = vehicle_info['type']
        vehicle_dir = self.data_dir / vehicle_type / vehicle_id
        
        logger.info(f"🚗 生成车辆数据: {vehicle_id} ({vehicle_type})")
        
        success_count = 0
        total_samples = len(self.states) * self.samples_per_state
        
        # 为每个状态生成音频
        for state in self.states:
            state_dir = vehicle_dir / state
            state_success = 0
            
            for sample_idx in range(self.samples_per_state):
                # 生成音频
                audio_data, filename = self.generate_vehicle_audio(
                    vehicle_id, vehicle_info, state, sample_idx
                )
                
                if audio_data is not None and filename is not None:
                    # 保存音频文件
                    file_path = state_dir / filename
                    if self.save_audio_file(audio_data, file_path):
                        state_success += 1
                        success_count += 1
            
            # 更新状态统计
            if state not in self.generation_stats['samples_by_state']:
                self.generation_stats['samples_by_state'][state] = 0
            self.generation_stats['samples_by_state'][state] += state_success
            
            logger.info(f"   {state}: {state_success}/{self.samples_per_state} 样本")
        
        # 更新类型统计
        if vehicle_type not in self.generation_stats['samples_by_type']:
            self.generation_stats['samples_by_type'][vehicle_type] = 0
        self.generation_stats['samples_by_type'][vehicle_type] += success_count
        
        success_rate = success_count / total_samples * 100
        logger.info(f"   完成度: {success_count}/{total_samples} ({success_rate:.1f}%)")
        
        return success_count == total_samples
    
    def generate_complete_dataset(self, vehicles_config: Dict = None, 
                                 force_regenerate: bool = False) -> bool:
        """生成完整的车辆个体识别数据集"""
        import time
        start_time = time.time()
        
        logger.info("🏭 开始生成车辆个体识别数据集")
        
        try:
            # 如果提供了车辆配置，先创建注册表
            if vehicles_config:
                if not self.create_vehicle_registry(vehicles_config):
                    return False
            
            # 加载车辆注册表
            if not self.label_manager.load_vehicle_registry():
                logger.error("❌ 无法加载车辆注册表")
                return False
            
            # 检查是否需要重新生成
            if not force_regenerate and self.data_dir.exists():
                logger.info("📁 数据目录已存在，检查完整性...")
                if self.label_manager.validate_dataset_structure():
                    logger.info("✅ 数据集已存在且完整，跳过生成")
                    return True
                else:
                    logger.info("⚠️ 数据集不完整，重新生成")
            
            # 初始化统计信息
            self.generation_stats = {
                'total_vehicles': len(self.label_manager.vehicle_to_label),
                'total_samples': 0,
                'samples_by_type': {},
                'samples_by_state': {},
                'generation_time': 0
            }
            
            # 生成每个车辆的数据
            successful_vehicles = 0
            total_vehicles = len(self.label_manager.vehicle_to_label)
            
            with tqdm(total=total_vehicles, desc="生成车辆数据") as pbar:
                for vehicle_id in self.label_manager.vehicle_to_label.keys():
                    vehicle_info = self.label_manager.get_vehicle_info(vehicle_id)
                    
                    if self.generate_vehicle_dataset(vehicle_id, vehicle_info):
                        successful_vehicles += 1
                    
                    pbar.update(1)
                    pbar.set_postfix({
                        '成功': f"{successful_vehicles}/{total_vehicles}",
                        '成功率': f"{successful_vehicles/total_vehicles*100:.1f}%"
                    })
            
            # 计算总样本数
            self.generation_stats['total_samples'] = sum(
                self.generation_stats['samples_by_state'].values()
            )
            
            # 记录生成时间
            self.generation_stats['generation_time'] = time.time() - start_time
            
            # 保存生成统计信息
            self.save_generation_stats()
            
            # 创建标签映射文件
            self.label_manager.create_label_mapping_file()
            
            # 打印最终统计
            self.print_generation_summary()
            
            success_rate = successful_vehicles / total_vehicles * 100
            logger.info(f"🎉 数据集生成完成! 成功率: {success_rate:.1f}%")
            
            return successful_vehicles == total_vehicles
            
        except Exception as e:
            logger.error(f"❌ 数据集生成失败: {str(e)}")
            return False
    
    def save_generation_stats(self):
        """保存生成统计信息"""
        try:
            stats_file = self.data_dir / "generation_stats.json"
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(self.generation_stats, f, indent=2, ensure_ascii=False)
            
            logger.info(f"📊 生成统计信息已保存: {stats_file}")
            
        except Exception as e:
            logger.error(f"❌ 保存统计信息失败: {str(e)}")
    
    def print_generation_summary(self):
        """打印生成摘要"""
        stats = self.generation_stats
        
        logger.info("📊 数据集生成摘要:")
        logger.info(f"   总车辆数: {stats['total_vehicles']}")
        logger.info(f"   总样本数: {stats['total_samples']}")
        logger.info(f"   生成时间: {stats['generation_time']:.1f} 秒")
        
        if stats['samples_by_type']:
            logger.info("   各类型样本分布:")
            for vehicle_type, count in stats['samples_by_type'].items():
                logger.info(f"     {vehicle_type}: {count} 样本")
        
        if stats['samples_by_state']:
            logger.info("   各状态样本分布:")
            for state, count in stats['samples_by_state'].items():
                logger.info(f"     {state}: {count} 样本")
        
        # 计算平均每车辆样本数
        if stats['total_vehicles'] > 0:
            avg_samples = stats['total_samples'] / stats['total_vehicles']
            logger.info(f"   平均每车辆样本数: {avg_samples:.1f}")
    
    def create_demo_dataset(self, num_vehicles_per_type: int = 3) -> bool:
        """创建演示用的小型数据集"""
        logger.info(f"🎭 创建演示数据集 (每类型 {num_vehicles_per_type} 辆车)")
        
        # 定义演示车辆配置
        demo_vehicles = {}
        vehicle_types = ['sedan', 'suv', 'truck', 'motorcycle', 'bus']
        
        for vehicle_type in vehicle_types:
            for i in range(num_vehicles_per_type):
                vehicle_id = f"{vehicle_type}_{i+1:02d}"
                demo_vehicles[vehicle_id] = {
                    'type': vehicle_type,
                    'brand': f"Brand_{vehicle_type.upper()}",
                    'model': f"Model_{i+1}",
                    'year': 2020 + i,
                    'engine_size': 2.0 + i * 0.5,
                    'description': f"演示用{vehicle_type}车辆 #{i+1}"
                }
        
        # 生成演示数据集
        return self.generate_complete_dataset(demo_vehicles, force_regenerate=True)
    
    def validate_generated_dataset(self) -> bool:
        """验证生成的数据集"""
        logger.info("🔍 验证生成的数据集...")
        
        # 验证数据集结构
        if not self.label_manager.validate_dataset_structure():
            return False
        
        # 验证音频文件质量
        logger.info("🎵 验证音频文件质量...")
        
        sample_count = 0
        error_count = 0
        
        for vehicle_id in list(self.label_manager.vehicle_to_label.keys())[:3]:  # 只检查前3个车辆
            vehicle_info = self.label_manager.get_vehicle_info(vehicle_id)
            vehicle_type = vehicle_info['type']
            vehicle_dir = self.data_dir / vehicle_type / vehicle_id
            
            for state in self.states:
                state_dir = vehicle_dir / state
                audio_files = list(state_dir.glob("*.wav"))
                
                for audio_file in audio_files[:2]:  # 每个状态只检查前2个文件
                    try:
                        audio_data, sr = sf.read(str(audio_file))
                        
                        # 检查音频参数
                        if sr != self.sample_rate:
                            logger.warning(f"⚠️ 采样率不匹配: {audio_file} ({sr} != {self.sample_rate})")
                            error_count += 1
                        
                        if len(audio_data) == 0:
                            logger.error(f"❌ 空音频文件: {audio_file}")
                            error_count += 1
                        
                        if np.max(np.abs(audio_data)) == 0:
                            logger.error(f"❌ 静音文件: {audio_file}")
                            error_count += 1
                        
                        sample_count += 1
                        
                    except Exception as e:
                        logger.error(f"❌ 读取音频文件失败 {audio_file}: {str(e)}")
                        error_count += 1
        
        error_rate = error_count / sample_count * 100 if sample_count > 0 else 0
        logger.info(f"📊 音频质量检查完成: {sample_count} 个样本, {error_count} 个错误 ({error_rate:.1f}%)")
        
        return error_rate < 10  # 错误率小于10%认为通过

def create_default_demo_config() -> Dict:
    """创建默认的演示配置"""
    return {
        # 轿车
        'sedan_01': {
            'type': 'sedan',
            'brand': 'Toyota',
            'model': 'Camry',
            'year': 2020,
            'engine_size': 2.5,
            'description': '中型轿车，平稳运行'
        },
        'sedan_02': {
            'type': 'sedan',
            'brand': 'Honda',
            'model': 'Accord',
            'year': 2021,
            'engine_size': 2.0,
            'description': '经济型轿车，低噪音'
        },
        'sedan_03': {
            'type': 'sedan',
            'brand': 'BMW',
            'model': '3 Series',
            'year': 2022,
            'engine_size': 3.0,
            'description': '豪华轿车，运动调校'
        },
        
        # SUV
        'suv_01': {
            'type': 'suv',
            'brand': 'Ford',
            'model': 'Explorer',
            'year': 2020,
            'engine_size': 3.5,
            'description': '中大型SUV，强劲动力'
        },
        'suv_02': {
            'type': 'suv',
            'brand': 'Jeep',
            'model': 'Cherokee',
            'year': 2021,
            'engine_size': 2.4,
            'description': '越野SUV，粗犷声音'
        },
        
        # 卡车
        'truck_01': {
            'type': 'truck',
            'brand': 'Ford',
            'model': 'F-150',
            'year': 2020,
            'engine_size': 5.0,
            'description': '皮卡车，低沉有力'
        },
        'truck_02': {
            'type': 'truck',
            'brand': 'Chevrolet',
            'model': 'Silverado',
            'year': 2021,
            'engine_size': 6.2,
            'description': '重型皮卡，深沉轰鸣'
        },
        
        # 摩托车
        'motorcycle_01': {
            'type': 'motorcycle',
            'brand': 'Harley-Davidson',
            'model': 'Street 750',
            'year': 2020,
            'engine_size': 0.75,
            'description': '巡航摩托，经典声浪'
        },
        'motorcycle_02': {
            'type': 'motorcycle',
            'brand': 'Yamaha',
            'model': 'YZF-R1',
            'year': 2021,
            'engine_size': 1.0,
            'description': '运动摩托，高频尖锐'
        }
    }

def main():
    """主函数 - 生成演示数据集"""
    logger.info("🚀 车辆个体识别数据集生成器")
    
    try:
        # 创建数据集生成器
        generator = IndividualDatasetGenerator()
        
        # 生成演示数据集
        demo_config = create_default_demo_config()
        
        if generator.generate_complete_dataset(demo_config, force_regenerate=True):
            logger.info("✅ 演示数据集生成成功")
            
            # 验证数据集
            if generator.validate_generated_dataset():
                logger.info("✅ 数据集验证通过")
            else:
                logger.warning("⚠️ 数据集验证发现问题")
        else:
            logger.error("❌ 演示数据集生成失败")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 程序执行失败: {str(e)}")
        return False

if __name__ == "__main__":
    main()