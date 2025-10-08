"""
相似度搜索公共工具模块

提供统一的特征处理、相似度计算和搜索结果处理功能，
避免在不同的相似度引擎中重复实现相同的逻辑。
"""

# 使用公共工具模块统一导入
from .common_utils import (
    os, sys, time, json, Path,
    List, Dict, Optional, Tuple, Any, Union,
    dataclass, asdict, warnings, np,
    config_manager, file_manager, logger_factory, TimerContext
)

import math
from typing import Callable
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.decomposition import PCA

from .exceptions import FeatureExtractionError, SimilaritySearchError

# 初始化日志器
similarity_logger = logger_factory.get_logger("相似度计算")


@dataclass
class VehicleProfile:
    """车辆档案数据结构"""
    vehicle_id: str
    features: np.ndarray
    metadata: Dict[str, Any]
    created_at: float = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = time.time()


@dataclass
class SearchResult:
    """搜索结果数据结构"""
    vehicle_id: str
    similarity_score: float
    metadata: Dict[str, Any]
    search_time: float = 0.0
    rank: int = 0


class FeatureProcessor:
    """特征处理工具类"""
    
    @staticmethod
    def normalize_features(features: np.ndarray) -> np.ndarray:
        """
        标准化特征向量
        
        Args:
            features: 输入特征向量
            
        Returns:
            标准化后的特征向量
        """
        if features is None or len(features) == 0:
            raise FeatureExtractionError("特征向量为空")
        
        # 确保特征是2D数组
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        # L2标准化
        return normalize(features, norm='l2')
    
    @staticmethod
    def build_feature_matrix(vehicle_profiles: Dict[str, VehicleProfile]) -> Tuple[np.ndarray, List[str]]:
        """
        构建特征矩阵
        
        Args:
            vehicle_profiles: 车辆档案字典
            
        Returns:
            特征矩阵和对应的车辆ID列表
        """
        if not vehicle_profiles:
            return None, []
        
        features_list = []
        vehicle_ids_list = []
        
        for vehicle_id, profile in vehicle_profiles.items():
            features_list.append(profile.features)
            vehicle_ids_list.append(vehicle_id)
        
        # 构建特征矩阵并标准化
        feature_matrix = np.vstack(features_list)
        feature_matrix = normalize(feature_matrix, norm='l2')
        
        return feature_matrix, vehicle_ids_list


class SimilarityCalculator:
    """相似度计算工具类"""
    
    @staticmethod
    def compute_cosine_similarity(query_features: np.ndarray, 
                                feature_matrix: np.ndarray) -> np.ndarray:
        """
        计算余弦相似度
        
        Args:
            query_features: 查询特征向量
            feature_matrix: 特征矩阵
            
        Returns:
            相似度数组
        """
        # 标准化查询特征
        query_features = FeatureProcessor.normalize_features(query_features)
        
        # 计算余弦相似度
        similarities = cosine_similarity(query_features, feature_matrix)[0]
        
        return similarities
    
    @staticmethod
    def filter_by_threshold(similarities: np.ndarray, 
                          vehicle_ids: List[str],
                          vehicle_profiles: Dict[str, VehicleProfile],
                          threshold: float,
                          top_k: int = None) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        根据阈值过滤搜索结果
        
        Args:
            similarities: 相似度数组
            vehicle_ids: 车辆ID列表
            vehicle_profiles: 车辆档案字典
            threshold: 相似度阈值
            top_k: 返回结果数量
            
        Returns:
            过滤后的搜索结果列表
        """
        # 获取相似度排序的索引
        sorted_indices = np.argsort(similarities)[::-1]
        
        # 构建结果 - 按车辆类型分组，每个类型只保留最高相似度的结果
        vehicle_type_best_scores = {}
        
        for i in sorted_indices:
            similarity = similarities[i]
            
            # 只处理超过阈值的结果
            if similarity >= threshold:
                full_vehicle_id = vehicle_ids[i]
                metadata = vehicle_profiles[full_vehicle_id].metadata
                
                # 提取车辆类型（从vehicle_id中提取，如bus_000, sedan_001等）
                vehicle_type = SimilarityCalculator._extract_vehicle_type(full_vehicle_id)
                
                # 如果这个车辆类型还没有记录，或者当前相似度更高，则更新
                if (vehicle_type not in vehicle_type_best_scores or 
                    similarity > vehicle_type_best_scores[vehicle_type][1]):
                    vehicle_type_best_scores[vehicle_type] = (full_vehicle_id, float(similarity), metadata)
        
        # 按相似度排序并返回前k个结果
        results = list(vehicle_type_best_scores.values())
        results.sort(key=lambda x: x[1], reverse=True)
        
        if top_k:
            results = results[:top_k]
        
        return results
    
    @staticmethod
    def _extract_base_vehicle_id(vehicle_id: str) -> str:
        """
        提取基础车辆ID（去掉音频文件的后缀标识）
        例如: bus_001_sample_1 -> bus_001
        """
        if '_' in vehicle_id:
            parts = vehicle_id.split('_')
            # 如果有超过2个部分，通常前两个部分是车辆类型和编号
            if len(parts) >= 2:
                return f"{parts[0]}_{parts[1]}"
            else:
                return parts[0]
        else:
            return vehicle_id
    
    @staticmethod
    def _extract_vehicle_type(vehicle_id: str) -> str:
        """
        从vehicle_id中提取车辆类型标识
        
        Args:
            vehicle_id: 完整的车辆ID，如 bus_002_bus_002_sample_000_d71beb90
            
        Returns:
            车辆类型，如 bus, sedan, suv, truck等
        """
        # 处理不同的ID格式
        if '_' in vehicle_id:
            parts = vehicle_id.split('_')
            # 查找车辆类型关键词
            vehicle_types = ['bus', 'sedan', 'suv', 'truck', 'motorcycle', 'my', 'random']
            
            # 查找车辆类型
            for part in parts:
                if part in vehicle_types:
                    return part
            # 如果没找到已知类型，返回第一个部分
            return parts[0]
        else:
            return vehicle_id
    
    @staticmethod
    def search_without_threshold(similarities: np.ndarray,
                               vehicle_ids: List[str],
                               vehicle_profiles: Dict[str, VehicleProfile],
                               top_k: int = None) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        不应用阈值的搜索（用于获取最佳匹配）
        按车辆类型分组，显示每种车辆类型的最佳匹配
        
        Args:
            similarities: 相似度数组
            vehicle_ids: 车辆ID列表
            vehicle_profiles: 车辆档案字典
            top_k: 返回结果数量
            
        Returns:
            搜索结果列表
        """
        # 获取相似度排序的索引
        sorted_indices = np.argsort(similarities)[::-1]
        
        # 构建结果 - 按车辆类型分组（显示不同车辆类型的最佳匹配）
        vehicle_type_best_scores = {}
        
        for i in sorted_indices:
            similarity = similarities[i]
            full_vehicle_id = vehicle_ids[i]
            metadata = vehicle_profiles[full_vehicle_id].metadata
            
            # 提取车辆类型（如bus, sedan, truck等）
            vehicle_type = SimilarityCalculator._extract_vehicle_type(full_vehicle_id)
            
            # 如果这个车辆类型还没有记录，或者当前相似度更高，则更新
            if (vehicle_type not in vehicle_type_best_scores or 
                similarity > vehicle_type_best_scores[vehicle_type][1]):
                vehicle_type_best_scores[vehicle_type] = (full_vehicle_id, float(similarity), metadata)
        
        # 按相似度排序并返回前k个结果
        results = list(vehicle_type_best_scores.values())
        results.sort(key=lambda x: x[1], reverse=True)
        
        # 限制返回结果数量
        if top_k:
            results = results[:top_k]
        
        return results


class SearchStatistics:
    """搜索统计工具类"""
    
    def __init__(self):
        self.stats = {
            'total_searches': 0,
            'successful_matches': 0,
            'avg_search_time': 0.0,
            'total_profiles': 0,
            'index_build_time': 0.0,
            'last_rebuild_time': 0.0
        }
    
    def update_search_stats(self, search_time: float, has_results: bool):
        """更新搜索统计信息"""
        self.stats['total_searches'] += 1
        if has_results:
            self.stats['successful_matches'] += 1
        
        # 更新平均搜索时间
        total_time = self.stats['avg_search_time'] * (self.stats['total_searches'] - 1) + search_time
        self.stats['avg_search_time'] = total_time / self.stats['total_searches']
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return self.stats.copy()
    
    def reset_stats(self):
        """重置统计信息"""
        self.stats = {
            'total_searches': 0,
            'successful_matches': 0,
            'avg_search_time': 0.0,
            'total_profiles': 0,
            'index_build_time': 0.0,
            'last_rebuild_time': 0.0
        }


from core.exceptions import handle_exceptions, safe_execute, log_system_error


def extract_and_validate_features(feature_extractor, audio_path: str) -> np.ndarray:
    """
    提取并验证音频特征
    
    Args:
        feature_extractor: 特征提取器实例
        audio_path: 音频文件路径
        
    Returns:
        提取的特征向量
        
    Raises:
        FeatureExtractionError: 特征提取失败
    """
    try:
        # 使用统一的extract_single接口
        result = feature_extractor.extract_single(audio_path)
        
        if not result.success or result.features is None:
            raise FeatureExtractionError(f"特征提取失败: {audio_path}")
        
        features = result.features
        
        # 处理字典格式的特征（从ECAPA提取器返回）
        if isinstance(features, dict):
            if 'fused_features' in features:
                features = features['fused_features']
            else:
                raise FeatureExtractionError(f"特征字典中缺少'fused_features'键: {audio_path}")
        
        if features is None or len(features) == 0:
            raise FeatureExtractionError(f"提取的特征为空: {audio_path}")
        
        # 确保特征是numpy数组
        if not isinstance(features, np.ndarray):
            features = np.array(features)
        
        return features
        
    except Exception as e:
        log_system_error(e, f"特征提取过程 - {audio_path}", include_traceback=True)
        raise FeatureExtractionError(f"特征提取过程中发生错误: {e}")