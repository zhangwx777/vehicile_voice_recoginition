# individual_label_manager.py
# 车辆个体识别标签管理器

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from core.logger import logger
from core.settings import PATH_CONFIG

class IndividualLabelManager:
    """车辆个体识别标签管理器"""
    
    def __init__(self, data_dir: str = None):
        self.data_dir = Path(data_dir or PATH_CONFIG['data_dir']) / "individual_vehicles"
        self.vehicle_to_label = {}  # vehicle_id -> numeric_label
        self.label_to_vehicle = {}  # numeric_label -> vehicle_id
        self.vehicle_info = {}      # vehicle_id -> vehicle_info
        self.num_vehicles = 0
        
    def load_vehicle_registry(self) -> bool:
        """加载车辆注册表"""
        registry_file = self.data_dir / "vehicle_registry.json"
        
        if not registry_file.exists():
            logger.error(f"❌ 车辆注册表文件不存在: {registry_file}")
            return False
        
        try:
            with open(registry_file, 'r', encoding='utf-8') as f:
                registry = json.load(f)
            
            # 按车辆ID排序以确保一致的标签分配
            sorted_vehicles = sorted(registry.keys())
            
            for idx, vehicle_id in enumerate(sorted_vehicles):
                self.vehicle_to_label[vehicle_id] = idx
                self.label_to_vehicle[idx] = vehicle_id
                self.vehicle_info[vehicle_id] = registry[vehicle_id]
            
            self.num_vehicles = len(sorted_vehicles)
            
            logger.info(f"✅ 成功加载 {self.num_vehicles} 辆车辆的标签映射")
            return True
            
        except Exception as e:
            logger.error(f"❌ 加载车辆注册表失败: {str(e)}")
            return False
    
    def get_vehicle_label(self, vehicle_id: str) -> Optional[int]:
        """获取车辆ID对应的数字标签"""
        return self.vehicle_to_label.get(vehicle_id)
    
    def get_vehicle_id(self, label: int) -> Optional[str]:
        """获取数字标签对应的车辆ID"""
        return self.label_to_vehicle.get(label)
    
    def get_vehicle_info(self, vehicle_id: str) -> Optional[Dict]:
        """获取车辆详细信息"""
        return self.vehicle_info.get(vehicle_id)
    
    def get_vehicles_by_type(self, vehicle_type: str) -> List[str]:
        """获取指定类型的所有车辆ID"""
        return [
            vehicle_id for vehicle_id, info in self.vehicle_info.items()
            if info['type'] == vehicle_type
        ]
    
    def create_label_mapping_file(self, output_path: str = None) -> Path:
        """创建标签映射文件（兼容现有系统）"""
        if not output_path:
            # 使用现有的models目录
            models_dir = Path(PATH_CONFIG['model_save_path']).parent
            output_path = models_dir / "individual_label_mapping.json"
        else:
            output_path = Path(output_path)
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 创建兼容格式的标签映射
        label_mapping = {
            'vehicle_to_label': self.vehicle_to_label,
            'label_to_vehicle': {str(k): v for k, v in self.label_to_vehicle.items()},
            'num_classes': self.num_vehicles,
            'mapping_type': 'individual_vehicle',
            'vehicle_info': self.vehicle_info
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(label_mapping, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ 个体标签映射文件已保存: {output_path}")
        return output_path
    
    def load_label_mapping_file(self, mapping_file: str) -> bool:
        """从文件加载标签映射"""
        mapping_path = Path(mapping_file)
        
        if not mapping_path.exists():
            logger.error(f"❌ 标签映射文件不存在: {mapping_path}")
            return False
        
        try:
            with open(mapping_path, 'r', encoding='utf-8') as f:
                mapping = json.load(f)
            
            self.vehicle_to_label = mapping['vehicle_to_label']
            self.label_to_vehicle = {int(k): v for k, v in mapping['label_to_vehicle'].items()}
            self.num_vehicles = mapping['num_classes']
            self.vehicle_info = mapping.get('vehicle_info', {})
            
            logger.info(f"✅ 成功加载标签映射: {self.num_vehicles} 个车辆")
            return True
            
        except Exception as e:
            logger.error(f"❌ 加载标签映射文件失败: {str(e)}")
            return False
    
    def get_dataset_statistics(self) -> Dict:
        """获取数据集统计信息"""
        stats = {
            'total_vehicles': self.num_vehicles,
            'vehicle_types': {},
            'vehicles_by_type': {}
        }
        
        for vehicle_id, info in self.vehicle_info.items():
            vehicle_type = info['type']
            if vehicle_type not in stats['vehicle_types']:
                stats['vehicle_types'][vehicle_type] = 0
                stats['vehicles_by_type'][vehicle_type] = []
            
            stats['vehicle_types'][vehicle_type] += 1
            stats['vehicles_by_type'][vehicle_type].append(vehicle_id)
        
        return stats
    
    def print_dataset_info(self):
        """打印数据集信息"""
        stats = self.get_dataset_statistics()
        
        logger.info("📊 车辆个体识别数据集信息:")
        logger.info(f"   总车辆数: {stats['total_vehicles']}")
        logger.info("   各类型车辆分布:")
        
        for vehicle_type, count in stats['vehicle_types'].items():
            logger.info(f"     {vehicle_type}: {count} 辆")
            for vehicle_id in stats['vehicles_by_type'][vehicle_type]:
                label = self.get_vehicle_label(vehicle_id)
                logger.info(f"       - {vehicle_id} (标签: {label})")
    
    def validate_dataset_structure(self) -> bool:
        """验证数据集结构完整性"""
        logger.info("🔍 验证车辆个体数据集结构...")
        
        if not self.data_dir.exists():
            logger.error(f"❌ 数据目录不存在: {self.data_dir}")
            return False
        
        missing_files = []
        states = ['idle', 'acceleration', 'cruising', 'deceleration']
        
        for vehicle_id in self.vehicle_to_label.keys():
            vehicle_info = self.get_vehicle_info(vehicle_id)
            vehicle_type = vehicle_info['type']
            
            vehicle_dir = self.data_dir / vehicle_type / vehicle_id
            
            if not vehicle_dir.exists():
                missing_files.append(f"车辆目录: {vehicle_dir}")
                continue
            
            for state in states:
                state_dir = vehicle_dir / state
                if not state_dir.exists():
                    missing_files.append(f"状态目录: {state_dir}")
                    continue
                
                # 检查音频文件
                audio_files = list(state_dir.glob("*.wav"))
                if len(audio_files) == 0:
                    missing_files.append(f"音频文件: {state_dir}/*.wav")
        
        if missing_files:
            logger.error("❌ 数据集结构验证失败，缺失文件:")
            for missing in missing_files[:10]:  # 只显示前10个
                logger.error(f"   - {missing}")
            if len(missing_files) > 10:
                logger.error(f"   ... 还有 {len(missing_files) - 10} 个缺失项")
            return False
        
        logger.info("✅ 数据集结构验证通过")
        return True
    
    def create_train_test_split(self, test_ratio: float = 0.2, 
                               random_seed: int = 42) -> Tuple[List[str], List[str]]:
        """创建训练/测试集分割（按车辆分割，确保同一车辆不会同时出现在训练和测试集中）"""
        np.random.seed(random_seed)
        
        all_vehicles = list(self.vehicle_to_label.keys())
        np.random.shuffle(all_vehicles)
        
        split_idx = int(len(all_vehicles) * (1 - test_ratio))
        train_vehicles = all_vehicles[:split_idx]
        test_vehicles = all_vehicles[split_idx:]
        
        logger.info(f"📊 数据集分割完成:")
        logger.info(f"   训练集: {len(train_vehicles)} 辆车辆")
        logger.info(f"   测试集: {len(test_vehicles)} 辆车辆")
        
        return train_vehicles, test_vehicles
    
    def get_class_weights(self) -> np.ndarray:
        """计算类别权重（用于处理不平衡数据）"""
        # 对于车辆个体识别，通常每个车辆的样本数相等
        # 但可以根据实际情况调整
        weights = np.ones(self.num_vehicles)
        return weights / np.sum(weights) * self.num_vehicles

def main():
    """测试标签管理器"""
    logger.info("🏷️ 车辆个体识别标签管理器测试")
    
    try:
        # 创建标签管理器
        label_manager = IndividualLabelManager()
        
        # 加载车辆注册表
        if not label_manager.load_vehicle_registry():
            logger.error("❌ 无法加载车辆注册表")
            return False
        
        # 打印数据集信息
        label_manager.print_dataset_info()
        
        # 验证数据集结构
        if not label_manager.validate_dataset_structure():
            logger.error("❌ 数据集结构验证失败")
            return False
        
        # 创建标签映射文件
        mapping_file = label_manager.create_label_mapping_file()
        
        # 创建训练/测试分割
        train_vehicles, test_vehicles = label_manager.create_train_test_split()
        
        logger.info("✅ 标签管理器测试完成")
        return True
        
    except Exception as e:
        logger.error(f"❌ 标签管理器测试失败: {str(e)}")
        return False

if __name__ == "__main__":
    main()