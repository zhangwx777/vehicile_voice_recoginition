import numpy as np
import torch
import pickle
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import time
from dataclasses import dataclass
from enhanced_feature_extractor import EnhancedAudioFeatureExtractor

@dataclass
class VehicleProfile:
    """车辆档案数据结构"""
    vehicle_id: str
    features: np.ndarray
    metadata: Dict[str, Any]
    timestamp: float
    
class VectorSimilarityEngine:
    """向量相似度搜索引擎"""
    
    def __init__(self, 
                 similarity_threshold: float = 0.7,
                 max_candidates: int = 10,
                 feature_extractor: Optional[EnhancedAudioFeatureExtractor] = None,
                 distance_metric: str = 'cosine',
                 index_type: str = 'flat',
                 search_params: Optional[Dict] = None,
                 **kwargs):
        """
        初始化相似度搜索引擎
        
        Args:
            similarity_threshold: 相似度阈值
            max_candidates: 最大候选数量
            feature_extractor: 特征提取器实例
        """
        self.similarity_threshold = similarity_threshold
        self.max_candidates = max_candidates
        self.distance_metric = distance_metric
        self.index_type = index_type
        self.search_params = search_params or {}
        
        # 初始化特征提取器
        if feature_extractor is None:
            self.feature_extractor = EnhancedAudioFeatureExtractor()
        else:
            self.feature_extractor = feature_extractor
        
        # 车辆档案数据库
        self.vehicle_profiles: Dict[str, VehicleProfile] = {}
        self.feature_matrix: Optional[np.ndarray] = None
        self.vehicle_ids: List[str] = []
        
        # 性能统计
        self.stats = {
            'total_searches': 0,
            'successful_matches': 0,
            'avg_search_time': 0.0,
            'total_profiles': 0
        }
    
    def add_vehicle_profile(self, 
                           vehicle_id: str, 
                           audio_path: str, 
                           metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        添加车辆档案
        
        Args:
            vehicle_id: 车辆ID
            audio_path: 音频文件路径
            metadata: 元数据
            
        Returns:
            bool: 是否成功添加
        """
        try:
            # 提取特征
            features_dict = self.feature_extractor.extract_features(audio_path)
            
            if not features_dict or 'fused_features' not in features_dict:
                pass  # 特征提取失败时静默处理
                return False
            
            features = features_dict['fused_features']
            if features is None or len(features) == 0:
                pass  # 特征为空时静默处理
                return False
            
            # 创建车辆档案
            profile = VehicleProfile(
                vehicle_id=vehicle_id,
                features=features,
                metadata=metadata or {},
                timestamp=time.time()
            )
            
            # 添加到数据库
            self.vehicle_profiles[vehicle_id] = profile
            self._rebuild_feature_matrix()
            
            self.stats['total_profiles'] = len(self.vehicle_profiles)
            # 成功添加车辆档案
            return True
            
        except Exception as e:
            pass  # 添加车辆档案失败时静默处理
            return False
    
    def _rebuild_feature_matrix(self):
        """重建特征矩阵"""
        if not self.vehicle_profiles:
            self.feature_matrix = None
            self.vehicle_ids = []
            return
        
        # 收集所有特征向量
        features_list = []
        vehicle_ids_list = []
        
        for vehicle_id, profile in self.vehicle_profiles.items():
            features_list.append(profile.features)
            vehicle_ids_list.append(vehicle_id)
        
        # 构建特征矩阵
        self.feature_matrix = np.vstack(features_list)
        self.vehicle_ids = vehicle_ids_list
        
        # 标准化特征矩阵
        self.feature_matrix = normalize(self.feature_matrix, norm='l2')
        
        # 特征矩阵重建完成
    
    def search_similar_vehicles(self, 
                               audio_path: str, 
                               top_k: Optional[int] = None) -> List[Tuple[str, float, Dict]]:
        """
        搜索相似车辆
        
        Args:
            audio_path: 查询音频路径
            top_k: 返回前k个结果
            
        Returns:
            List[Tuple[str, float, Dict]]: (vehicle_id, similarity, metadata)
        """
        start_time = time.time()
        
        try:
            if self.feature_matrix is None or len(self.vehicle_profiles) == 0:
                return []
            
            # 提取查询音频特征
            query_features_dict = self.feature_extractor.extract_features(audio_path)
            
            if not query_features_dict or 'fused_features' not in query_features_dict:
                return []
            
            query_features = query_features_dict['fused_features']
            if query_features is None or len(query_features) == 0:
                return []
            
            # 标准化查询特征
            query_features = normalize(query_features.reshape(1, -1), norm='l2')
            
            # 计算余弦相似度
            similarities = cosine_similarity(query_features, self.feature_matrix)[0]
            
            # 获取相似度排序的索引
            sorted_indices = np.argsort(similarities)[::-1]
            
            # 构建结果
            results = []
            k = top_k or self.max_candidates
            
            for i in sorted_indices[:k]:
                similarity = similarities[i]
                
                # 只返回超过阈值的结果
                if similarity >= self.similarity_threshold:
                    vehicle_id = self.vehicle_ids[i]
                    metadata = self.vehicle_profiles[vehicle_id].metadata
                    results.append((vehicle_id, float(similarity), metadata))
            
            # 更新统计信息
            search_time = time.time() - start_time
            self.stats['total_searches'] += 1
            if results:
                self.stats['successful_matches'] += 1
            
            # 更新平均搜索时间
            total_time = self.stats['avg_search_time'] * (self.stats['total_searches'] - 1) + search_time
            self.stats['avg_search_time'] = total_time / self.stats['total_searches']
            
            return results
            
        except Exception as e:
            pass  # 相似度搜索失败时静默处理
            return []
    
    def get_vehicle_info(self, vehicle_id: str) -> Optional[Dict[str, Any]]:
        """获取车辆信息"""
        if vehicle_id in self.vehicle_profiles:
            profile = self.vehicle_profiles[vehicle_id]
            return {
                'vehicle_id': profile.vehicle_id,
                'metadata': profile.metadata,
                'timestamp': profile.timestamp,
                'feature_dim': len(profile.features)
            }
        return None
    
    def remove_vehicle_profile(self, vehicle_id: str) -> bool:
        """删除车辆档案"""
        if vehicle_id in self.vehicle_profiles:
            del self.vehicle_profiles[vehicle_id]
            self._rebuild_feature_matrix()
            self.stats['total_profiles'] = len(self.vehicle_profiles)
            # 删除车辆档案
            return True
        return False
    
    def save_database(self, save_path: str) -> bool:
        """保存数据库"""
        try:
            save_data = {
                'vehicle_profiles': {},
                'stats': self.stats,
                'config': {
                    'similarity_threshold': self.similarity_threshold,
                    'max_candidates': self.max_candidates
                }
            }
            
            # 序列化车辆档案
            for vehicle_id, profile in self.vehicle_profiles.items():
                save_data['vehicle_profiles'][vehicle_id] = {
                    'vehicle_id': profile.vehicle_id,
                    'features': profile.features.tolist(),
                    'metadata': profile.metadata,
                    'timestamp': profile.timestamp
                }
            
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, ensure_ascii=False, indent=2)
            
            # 数据库已保存
            return True
            
        except Exception as e:
            pass  # 保存数据库失败时静默处理
            return False
    
    def load_database(self, load_path: str) -> bool:
        """加载数据库"""
        try:
            if not Path(load_path).exists():
                pass  # 数据库文件不存在时静默处理
                return False
            
            with open(load_path, 'r', encoding='utf-8') as f:
                save_data = json.load(f)
            
            # 恢复配置
            config = save_data.get('config', {})
            self.similarity_threshold = config.get('similarity_threshold', self.similarity_threshold)
            self.max_candidates = config.get('max_candidates', self.max_candidates)
            
            # 恢复统计信息
            self.stats = save_data.get('stats', self.stats)
            
            # 恢复车辆档案
            self.vehicle_profiles = {}
            profiles_data = save_data.get('vehicle_profiles', {})
            
            for vehicle_id, profile_data in profiles_data.items():
                profile = VehicleProfile(
                    vehicle_id=profile_data['vehicle_id'],
                    features=np.array(profile_data['features']),
                    metadata=profile_data['metadata'],
                    timestamp=profile_data['timestamp']
                )
                self.vehicle_profiles[vehicle_id] = profile
            
            # 重建特征矩阵
            self._rebuild_feature_matrix()
            
            # 数据库已加载
            return True
            
        except Exception as e:
            pass  # 加载数据库失败时静默处理
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            **self.stats,
            'database_size': len(self.vehicle_profiles),
            'feature_matrix_shape': self.feature_matrix.shape if self.feature_matrix is not None else None,
            'similarity_threshold': self.similarity_threshold,
            'max_candidates': self.max_candidates
        }
    
    def clear_database(self):
        """清空数据库"""
        self.vehicle_profiles = {}
        self.feature_matrix = None
        self.vehicle_ids = []
        self.stats = {
            'total_searches': 0,
            'successful_matches': 0,
            'avg_search_time': 0.0,
            'total_profiles': 0
        }
        # 数据库已清空

def test_similarity_engine():
    """测试相似度搜索引擎"""
    print("=== 测试向量相似度搜索引擎 ===")
    
    # 创建搜索引擎
    engine = VectorSimilarityEngine(
        similarity_threshold=0.5,
        max_candidates=5
    )
    
    # 测试音频文件路径（需要根据实际情况调整）
    test_audio = "test_audio.wav"
    
    # 创建测试音频文件（生成简单的正弦波）
    import soundfile as sf
    sample_rate = 16000
    duration = 3.0
    frequency = 440  # A4音符
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio_data = 0.3 * np.sin(2 * np.pi * frequency * t)
    
    try:
        sf.write(test_audio, audio_data, sample_rate)
        print(f"创建测试音频文件: {test_audio}")
        
        # 添加测试车辆档案
        success = engine.add_vehicle_profile(
            vehicle_id="TEST_001",
            audio_path=test_audio,
            metadata={"brand": "测试品牌", "model": "测试型号"}
        )
        
        if success:
            print("车辆档案添加成功")
            
            # 测试搜索
            results = engine.search_similar_vehicles(test_audio, top_k=3)
            print(f"搜索结果: {results}")
            
            # 显示统计信息
            stats = engine.get_stats()
            print(f"引擎统计: {stats}")
        else:
            print("车辆档案添加失败")
            
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
    finally:
        # 清理测试文件
        if Path(test_audio).exists():
            Path(test_audio).unlink()
            print(f"清理测试文件: {test_audio}")

if __name__ == "__main__":
    test_similarity_engine()