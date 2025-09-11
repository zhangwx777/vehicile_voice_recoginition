#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
车辆声纹识别统一引擎
整合分类模式、相似度搜索模式和混合模式
"""

import os
import json
import time
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import logging

# 导入配置
from config import (
    PATH_CONFIG, ENHANCED_FEATURE_CONFIG,
    SIMILARITY_CONFIG, SYSTEM_MODE_CONFIG, AUDIO_CONFIG
)

# 先导入logger
from core.logger import logger
from data.preprocessor import AudioPreprocessor

# 导入增强功能模块
try:
    from enhanced_feature_extractor import EnhancedAudioFeatureExtractor
    from vector_similarity_engine import VectorSimilarityEngine, VehicleProfile
except ImportError as e:
    logger.warning(f"增强功能模块导入失败: {e}")
    EnhancedAudioFeatureExtractor = None
    VectorSimilarityEngine = None
    VehicleProfile = None

class RecognitionMode(Enum):
    """识别模式枚举"""
    SIMILARITY = "similarity"

@dataclass
class AudioInfo:
    """音频信息数据类"""
    file_path: str
    file_name: str
    file_size: int
    duration: Optional[float] = None
    sample_rate: Optional[int] = None
    channels: Optional[int] = None

@dataclass
class RecognitionResult:
    """识别结果数据结构"""
    vehicle_id: str
    confidence: float
    method: str
    metadata: Dict[str, Any]
    processing_time: float
    
    # 分类结果特有字段
    class_probabilities: Optional[Dict[str, float]] = None
    
    # 相似度结果特有字段
    similarity_score: Optional[float] = None
    similar_vehicles: Optional[List[Tuple[str, float]]] = None

class UnifiedVehicleRecognitionEngine:
    """统一车辆声纹识别引擎"""
    
    def __init__(self, 
                 mode: Union[str, RecognitionMode] = RecognitionMode.SIMILARITY,
                 config_override: Optional[Dict] = None):
        """
        初始化统一识别引擎
        
        Args:
            mode: 识别模式（仅支持相似度搜索）
            config_override: 配置覆盖
        """
        # 设置识别模式
        if isinstance(mode, str):
            self.mode = RecognitionMode(mode)
        else:
            self.mode = mode
            
        # 初始化车辆声纹识别引擎
        
        # 合并配置
        self.config = self._merge_config(config_override)
        
        # 初始化组件
        self.audio_preprocessor = None
        self.enhanced_extractor = None
        self.similarity_engine = None
        
        # 性能统计
        self.stats = {
            'total_recognitions': 0,
            'successful_recognitions': 0,
            'avg_processing_time': 0.0,
            'mode_usage': {
                'similarity': 0
            }
        }
        
        # 根据模式初始化相应组件
        self._initialize_components()
    
    def _merge_config(self, config_override: Optional[Dict]) -> Dict:
        """合并配置"""
        config = {
            'enhanced_feature': ENHANCED_FEATURE_CONFIG,
            'similarity': SIMILARITY_CONFIG,
            'system_mode': SYSTEM_MODE_CONFIG,
            'audio': AUDIO_CONFIG,
            'paths': PATH_CONFIG
        }
        
        if config_override:
            for key, value in config_override.items():
                if key in config and isinstance(config[key], dict):
                    config[key].update(value)
                else:
                    config[key] = value
        
        return config
    
    def _initialize_components(self):
        """根据模式初始化组件"""
        try:
            # 初始化音频预处理器 - 过滤不支持的参数
            audio_config = self.config['audio'].copy()
            # 移除AudioPreprocessor不支持的参数
            unsupported_params = ['win_length', 'fmin', 'fmax', 'normalize', 'to_db', 'ref', 'amin', 'top_db', 'augmentation', 'duration']
            for param in unsupported_params:
                audio_config.pop(param, None)
            
            self.audio_preprocessor = AudioPreprocessor(**audio_config)
            # 音频预处理器初始化完成
            
            # 初始化相似度搜索组件
            self._initialize_similarity_components()
                
        except Exception as e:
            logger.error(f"组件初始化失败: {e}")
            raise
    

    
    def _initialize_similarity_components(self):
        """初始化相似度搜索组件"""
        try:
            if EnhancedAudioFeatureExtractor is None or VectorSimilarityEngine is None:
                logger.error("增强功能模块不可用，无法初始化相似度搜索组件")
                self.enhanced_extractor = None
                self.similarity_engine = None
                return
            
            # 初始化增强特征提取器
            self.enhanced_extractor = EnhancedAudioFeatureExtractor(
                **self.config['enhanced_feature']
            )
            # 增强特征提取器初始化完成
            
            # 初始化相似度搜索引擎
            self.similarity_engine = VectorSimilarityEngine(
                feature_extractor=self.enhanced_extractor,
                **self.config['similarity']
            )
            
            # 尝试加载现有的车辆数据库
            database_path = self.config['paths'].get('vector_database_path')
            if database_path and os.path.exists(database_path):
                self.similarity_engine.load_database(database_path)
                # 车辆向量数据库加载完成
            
            # 相似度搜索引擎初始化完成
            
        except Exception as e:
            logger.error(f"相似度组件初始化失败: {e}")
            self.enhanced_extractor = None
            self.similarity_engine = None
    
    def recognize(self, audio_path: str, **kwargs) -> Tuple[RecognitionResult, AudioInfo]:
        """
        识别车辆
        
        Args:
            audio_path: 音频文件路径
            **kwargs: 额外参数
            
        Returns:
            Tuple[RecognitionResult, AudioInfo]: 识别结果和音频信息
        """
        start_time = time.time()
        
        try:
            # 验证音频文件
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"音频文件不存在: {audio_path}")
            
            # 使用相似度搜索进行识别
            result = self._search_similar_vehicle(audio_path, **kwargs)
            
            # 更新统计信息
            processing_time = time.time() - start_time
            result.processing_time = processing_time
            
            self._update_stats(result, processing_time)
            
            # 创建音频信息
            audio_info = self._get_audio_info(audio_path)
            
            return result, audio_info
            
        except Exception as e:
            logger.error(f"车辆识别失败: {e}")
            processing_time = time.time() - start_time
            
            # 创建音频信息
            audio_info = self._get_audio_info(audio_path)
            
            return RecognitionResult(
                vehicle_id="unknown",
                confidence=0.0,
                method="error",
                metadata={"error": str(e)},
                processing_time=processing_time
            ), audio_info
    

    
    def _search_similar_vehicle(self, audio_path: str, **kwargs) -> RecognitionResult:
        """使用相似度搜索识别车辆"""
        if self.similarity_engine is None:
            # 相似度搜索引擎未初始化
            return RecognitionResult(
                vehicle_id="unknown",
                confidence=0.0,
                method="similarity",
                metadata={"error": "相似度搜索引擎未初始化"},
                processing_time=0.0,
                similarity_score=0.0,
                similar_vehicles=[]
            )
        
        # 搜索相似车辆
        top_k = kwargs.get('top_k', self.config['similarity']['max_candidates'])
        similar_vehicles = self.similarity_engine.search_similar_vehicles(audio_path, top_k)
        
        if not similar_vehicles:
            return RecognitionResult(
                vehicle_id="unknown",
                confidence=0.0,
                method="similarity",
                metadata={"reason": "no_similar_vehicles_found"},
                processing_time=0.0,
                similarity_score=0.0,
                similar_vehicles=[]
            )
        
        # 获取最相似的车辆
        best_match = similar_vehicles[0]
        vehicle_id, similarity_score, metadata = best_match
        
        return RecognitionResult(
            vehicle_id=vehicle_id,
            confidence=similarity_score,
            method="similarity",
            metadata={
                "feature_extractor": "enhanced",
                "search_results_count": len(similar_vehicles),
                "vehicle_metadata": metadata
            },
            processing_time=0.0,
            similarity_score=similarity_score,
            similar_vehicles=[(vid, score) for vid, score, _ in similar_vehicles]
        )
    
    def add_vehicle_profile(self, vehicle_id: str, audio_path: str, metadata: Dict[str, Any]) -> bool:
        """添加车辆档案"""
        if self.similarity_engine is None:
            # 相似度搜索引擎未初始化
            return False
        
        try:
            success = self.similarity_engine.add_vehicle_profile(vehicle_id, audio_path, metadata)
            if success:
                # 车辆档案添加成功
                pass
            return success
        except Exception as e:
            logger.error(f"添加车辆档案失败: {e}")
            return False
    
    def save_database(self, path: Optional[str] = None) -> bool:
        """保存车辆数据库"""
        if self.similarity_engine is None:
            return False
        
        if path is None:
            path = self.config['paths'].get('vector_database_path')
        
        try:
            return self.similarity_engine.save_database(path)
        except Exception as e:
            logger.error(f"保存数据库失败: {e}")
            return False
    
    def load_database(self, path: Optional[str] = None) -> bool:
        """加载车辆数据库"""
        if self.similarity_engine is None:
            return False
        
        if path is None:
            path = self.config['paths'].get('vector_database_path')
        
        try:
            return self.similarity_engine.load_database(path)
        except Exception as e:
            logger.error(f"加载数据库失败: {e}")
            return False
    
    def _get_audio_info(self, audio_path: str) -> AudioInfo:
        """
        获取音频文件信息
        
        Args:
            audio_path: 音频文件路径
            
        Returns:
            AudioInfo: 音频信息对象
        """
        try:
            import librosa
            
            # 获取文件基本信息
            file_path = Path(audio_path)
            file_size = file_path.stat().st_size if file_path.exists() else 0
            
            # 获取音频信息
            try:
                y, sr = librosa.load(audio_path, sr=None)
                duration = len(y) / sr
                sample_rate = sr
                channels = 1 if len(y.shape) == 1 else y.shape[0]
            except:
                duration = None
                sample_rate = None
                channels = None
            
            return AudioInfo(
                file_path=str(file_path),
                file_name=file_path.name,
                file_size=file_size,
                duration=duration,
                sample_rate=sample_rate,
                channels=channels
            )
            
        except Exception as e:
            logger.warning(f"获取音频信息失败: {e}")
            file_path = Path(audio_path)
            return AudioInfo(
                file_path=str(file_path),
                file_name=file_path.name,
                file_size=file_path.stat().st_size if file_path.exists() else 0
            )
    
    def _update_stats(self, result: RecognitionResult, processing_time: float):
        """更新统计信息"""
        self.stats['total_recognitions'] += 1
        
        if result.vehicle_id != "unknown":
            self.stats['successful_recognitions'] += 1
        
        # 更新平均处理时间
        total_time = self.stats['avg_processing_time'] * (self.stats['total_recognitions'] - 1)
        self.stats['avg_processing_time'] = (total_time + processing_time) / self.stats['total_recognitions']
        
        # 更新模式使用统计
        if 'hybrid' in result.method:
            self.stats['mode_usage']['hybrid'] += 1
        elif 'classification' in result.method:
            self.stats['mode_usage']['classification'] += 1
        elif 'similarity' in result.method:
            self.stats['mode_usage']['similarity'] += 1
    
    def get_engine_stats(self) -> Dict[str, Any]:
        """获取引擎统计信息（兼容方法）"""
        return self.get_stats()
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = self.stats.copy()
        
        # 添加成功率
        if stats['total_recognitions'] > 0:
            stats['success_rate'] = stats['successful_recognitions'] / stats['total_recognitions']
        else:
            stats['success_rate'] = 0.0
        
        # 添加组件状态
        stats['components_status'] = {
            'enhanced_extractor': self.enhanced_extractor is not None,
            'similarity_engine': self.similarity_engine is not None,
            'audio_preprocessor': self.audio_preprocessor is not None
        }
        
        return stats
    
    def get_system_info(self) -> Dict[str, Any]:
        """获取系统信息"""
        info = {
            'mode': self.mode.value,
            'config': self.config,
            'stats': self.get_stats(),
            'components': {
                'cnn_model': {
                    'available': self.cnn_model is not None,
                    'num_classes': len(self.label_mapping) if self.label_mapping else 0
                },
                'similarity_engine': {
                    'available': self.similarity_engine is not None,
                    'num_profiles': len(self.similarity_engine.vehicle_profiles) if self.similarity_engine else 0
                }
            }
        }
        
        return info