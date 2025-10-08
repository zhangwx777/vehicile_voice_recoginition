#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高级向量相似度搜索引擎
集成FAISS向量数据库，大幅提升搜索性能和扩展性
"""

# 使用公共工具模块统一导入
from core.common_utils import (
    os, sys, time, json, Path,
    List, Dict, Optional, Tuple, Any, Union,
    dataclass, asdict, warnings, np,
    config_manager, file_manager, logger_factory, TimerContext
)

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

# FAISS导入和可用性检查
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    faiss = None

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.exceptions import handle_exceptions, SimilaritySearchError
from core.similarity_utils import (
    VehicleProfile, SearchResult, FeatureProcessor, 
    SimilarityCalculator, SearchStatistics, extract_and_validate_features
)

# 导入优化特征提取器
from optimized_feature_extractor import create_optimized_extractor


class AdvancedSimilarityEngine:
    """高级向量相似度搜索引擎"""
    
    def __init__(self, 
                 feature_dim: int = 192,
                 similarity_threshold: float = 0.7,
                 max_candidates: int = 10,
                 index_type: str = 'auto',
                 use_gpu: bool = False,
                 n_probe: int = 10,
                 **kwargs):
        """
        初始化高级相似度搜索引擎
        
        Args:
            feature_dim: 特征维度
            similarity_threshold: 相似度阈值
            max_candidates: 最大候选数量
            index_type: 索引类型 ('flat', 'ivf', 'hnsw', 'auto')
            use_gpu: 是否使用GPU加速
            n_probe: IVF索引探测数量
        """
        self.feature_dim = feature_dim
        self.similarity_threshold = similarity_threshold
        self.max_candidates = max_candidates
        self.index_type = index_type
        self.use_gpu = use_gpu and faiss.get_num_gpus() > 0 if FAISS_AVAILABLE else False
        self.n_probe = n_probe
        
        # 使用公共工具模块的日志器
        self.logger = logger_factory.get_logger("高级相似度引擎")
        
        # 车辆档案存储
        self.vehicle_profiles: Dict[str, VehicleProfile] = {}
        self.vehicle_ids: List[str] = []
        
        # FAISS索引
        self.faiss_index = None
        self.index_trained = False
        
        # 特征提取器 - 使用优化版本
        self.feature_extractor = create_optimized_extractor(
            use_cache=True,
            max_workers=2  # 适中的线程数
        )
        self.logger.info("使用优化的特征提取器")
        
        # 使用公共统计工具
        self.stats_manager = SearchStatistics()
        
        # 线程锁
        self._lock = threading.RLock()
        
        self.logger.info(f"初始化高级相似度引擎 - 特征维度: {feature_dim}, 索引类型: {index_type}")
        
        # 初始化FAISS索引
        self._initialize_faiss_index()
    
    def _initialize_faiss_index(self):
        """初始化FAISS索引"""
        if not FAISS_AVAILABLE:
            self.logger.warning("FAISS不可用，使用基础相似度搜索")
            return
        
        try:
            # 根据数据规模自动选择索引类型
            if self.index_type == 'auto':
                self.index_type = self._auto_select_index_type()
            
            # 创建索引 - 使用L2距离而非内积，以便与余弦相似度保持一致
            if self.index_type == 'flat':
                # 精确搜索，适合小规模数据
                self.faiss_index = faiss.IndexFlatL2(self.feature_dim)
                
            elif self.index_type == 'ivf':
                # 倒排文件索引，适合中等规模数据
                n_centroids = min(100, max(10, len(self.vehicle_profiles) // 10))
                quantizer = faiss.IndexFlatL2(self.feature_dim)
                self.faiss_index = faiss.IndexIVFFlat(quantizer, self.feature_dim, n_centroids)
                self.faiss_index.nprobe = self.n_probe
                
            elif self.index_type == 'hnsw':
                # 分层导航小世界图，适合大规模数据
                self.faiss_index = faiss.IndexHNSWFlat(self.feature_dim, 32)
                self.faiss_index.hnsw.efConstruction = 200
                self.faiss_index.hnsw.efSearch = 50
                
            else:
                # 默认使用Flat索引
                self.faiss_index = faiss.IndexFlatL2(self.feature_dim)
            
            # GPU加速
            if self.use_gpu and FAISS_AVAILABLE:
                try:
                    res = faiss.StandardGpuResources()
                    self.faiss_index = faiss.index_cpu_to_gpu(res, 0, self.faiss_index)
                    self.logger.info("启用GPU加速")
                except Exception as e:
                    self.logger.warning(f"GPU加速启用失败，使用CPU: {e}")
                    self.use_gpu = False
            
            self.logger.info(f"FAISS索引初始化完成 - 类型: {self.index_type}")
            
        except Exception as e:
            self.logger.error(f"FAISS索引初始化失败: {e}")
            self.faiss_index = None
    
    def _auto_select_index_type(self) -> str:
        """根据数据规模自动选择索引类型"""
        n_profiles = len(self.vehicle_profiles)
        
        if n_profiles < 100:
            return 'flat'  # 小规模，精确搜索
        elif n_profiles < 10000:
            return 'ivf'   # 中等规模，倒排索引
        else:
            return 'hnsw'  # 大规模，图索引
    
    @handle_exceptions(default_return=False, log_error=True)
    def add_vehicle_profile(self, 
                           vehicle_id: str, 
                           audio_path: str = None,
                           features: np.ndarray = None,
                           metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        添加车辆档案
        
        Args:
            vehicle_id: 车辆ID
            audio_path: 音频文件路径（如果features为None则需要提供）
            features: 已提取的特征（如果提供则不会重新提取）
            metadata: 元数据
            
        Returns:
            bool: 是否成功添加
        """
        with self._lock:
            try:
                self.logger.info(f"开始添加车辆档案: {vehicle_id}")
                
                # 如果没有提供特征，则从音频文件提取
                if features is None:
                    if audio_path is None:
                        self.logger.error("未提供音频路径或特征")
                        raise ValueError("必须提供audio_path或features参数")
                    self.logger.debug(f"从音频文件提取特征: {audio_path}")
                    features = extract_and_validate_features(self.feature_extractor, audio_path)
                    self.logger.debug(f"特征提取完成，维度: {features.shape}")
                else:
                    self.logger.debug(f"使用提供的特征，维度: {features.shape}")
                
                # 调试信息：记录特征统计
                self.logger.debug(f"特征统计 - 最小值: {np.min(features):.4f}, 最大值: {np.max(features):.4f}, 均值: {np.mean(features):.4f}, 标准差: {np.std(features):.4f}")
                
                # 使用公共工具标准化特征
                processor = FeatureProcessor()
                features = processor.normalize_features(features)
                self.logger.debug(f"特征标准化完成，维度: {features.shape}")
                
                # 调试信息：记录标准化后特征统计
                self.logger.debug(f"标准化后特征统计 - 最小值: {np.min(features):.4f}, 最大值: {np.max(features):.4f}, 均值: {np.mean(features):.4f}, 标准差: {np.std(features):.4f}")
                
                # 创建车辆档案
                profile = VehicleProfile(
                    vehicle_id=vehicle_id,
                    features=features,
                    metadata=metadata or {},
                    created_at=time.time()
                )
                self.logger.debug(f"车辆档案创建完成: {vehicle_id}")
                
                # 检查是否已存在
                if vehicle_id in self.vehicle_profiles:
                    self.logger.warning(f"车辆档案已存在，将更新: {vehicle_id}")
                    self._remove_from_index(vehicle_id)
                
                # 添加到内存
                self.vehicle_profiles[vehicle_id] = profile
                if vehicle_id not in self.vehicle_ids:
                    self.vehicle_ids.append(vehicle_id)
                self.logger.debug(f"车辆档案已添加到内存: {vehicle_id}")
                
                # 添加到FAISS索引
                self.logger.debug(f"开始将车辆档案添加到FAISS索引: {vehicle_id}")
                self._add_to_faiss_index(profile)
                self.logger.debug(f"车辆档案已添加到FAISS索引: {vehicle_id}")
                
                # 调试信息：记录FAISS索引状态
                if FAISS_AVAILABLE and self.faiss_index is not None:
                    self.logger.debug(f"FAISS索引状态 - 类型: {self.index_type}, 总数: {self.faiss_index.ntotal}, 已训练: {self.index_trained}")
                
                # 更新统计
                self.stats_manager.stats['total_profiles'] = len(self.vehicle_profiles)
                self.logger.debug(f"统计信息已更新，总档案数: {len(self.vehicle_profiles)}")
                
                self.logger.info(f"车辆档案添加成功: {vehicle_id}")
                return True
                
            except Exception as e:
                self.logger.error(f"添加车辆档案失败 {vehicle_id}: {e}")
                return False
    
    def _add_to_faiss_index(self, profile: VehicleProfile):
        """添加特征到FAISS索引"""
        if not FAISS_AVAILABLE or self.faiss_index is None:
            return
        
        try:
            self.logger.debug(f"开始添加到FAISS索引: {profile.vehicle_id}")
            
            features = profile.features.reshape(1, -1).astype(np.float32)
            self.logger.debug(f"特征向量重塑完成，形状: {features.shape}")
            self.logger.debug(f"准备添加到FAISS索引 - 车辆ID: {profile.vehicle_id}, 特征形状: {features.shape}")
            
            # 对于需要训练的索引类型
            if self.index_type == 'ivf' and not self.index_trained:
                self.logger.debug("IVF索引未训练，开始训练索引")
                self._train_index()
            
            self.logger.debug(f"正在将特征添加到FAISS索引...")
            self.faiss_index.add(features)
            self.logger.debug(f"特征已成功添加到FAISS索引，当前总数: {self.faiss_index.ntotal}")
            self.logger.debug(f"特征向量已添加到FAISS索引: {profile.vehicle_id}")
            
        except Exception as e:
            self.logger.error(f"添加到FAISS索引失败: {e}")
    
    def _train_index(self):
        """训练FAISS索引（仅IVF类型需要）"""
        if not FAISS_AVAILABLE or self.faiss_index is None:
            return
        
        if self.index_type != 'ivf' or self.index_trained:
            return
        
        try:
            self.logger.info("开始训练FAISS索引")
            
            # 收集所有特征用于训练
            all_features = []
            for profile in self.vehicle_profiles.values():
                all_features.append(profile.features)
            
            if len(all_features) < 10:  # 需要足够的训练数据
                self.logger.warning(f"训练数据不足，当前只有{len(all_features)}个特征，至少需要10个")
                return
            
            training_data = np.vstack(all_features).astype(np.float32)
            self.logger.debug(f"收集到{len(all_features)}个特征用于训练，数据形状: {training_data.shape}")
            
            self.faiss_index.train(training_data)
            self.index_trained = True
            self.logger.debug("FAISS索引训练完成")
            
        except Exception as e:
            self.logger.error(f"FAISS索引训练失败: {e}")
    
    def _remove_from_index(self, vehicle_id: str):
        """从索引中移除车辆（FAISS不支持直接删除，需要重建）"""
        if vehicle_id in self.vehicle_profiles:
            del self.vehicle_profiles[vehicle_id]
            if vehicle_id in self.vehicle_ids:
                self.vehicle_ids.remove(vehicle_id)
            
            # 重建索引
            self._rebuild_faiss_index()
    
    def _rebuild_faiss_index(self):
        """重建FAISS索引"""
        if not FAISS_AVAILABLE:
            return
        
        start_time = time.time()
        
        try:
            # 重新初始化索引
            self._initialize_faiss_index()
            
            # 重新添加所有特征
            if self.faiss_index is not None:
                all_features = []
                for profile in self.vehicle_profiles.values():
                    all_features.append(profile.features)
                
                if all_features:
                    features_matrix = np.vstack(all_features).astype(np.float32)
                    
                    # 训练索引（如果需要）
                    if self.index_type == 'ivf':
                        self.faiss_index.train(features_matrix)
                        self.index_trained = True
                    
                    # 添加特征
                    self.faiss_index.add(features_matrix)
            
            rebuild_time = time.time() - start_time
            self.stats_manager.stats['index_build_time'] = rebuild_time
            self.stats_manager.stats['last_rebuild_time'] = time.time()
            
            self.logger.info(f"FAISS索引重建完成，耗时: {rebuild_time:.3f}s")
            
            # 重建后自动保存索引
            self._save_faiss_index()
            
        except Exception as e:
            self.logger.error(f"FAISS索引重建失败: {e}")
    
    def _save_faiss_index(self):
        """保存FAISS索引到文件"""
        if not FAISS_AVAILABLE or self.faiss_index is None:
            return
        
        try:
            import tempfile
            import shutil
            import os
            
            # 确保models目录存在
            os.makedirs('models', exist_ok=True)
            
            # 使用系统临时目录（通常是英文路径）创建临时文件
            temp_dir = tempfile.gettempdir()  # 获取系统临时目录
            temp_filename = f"faiss_index_{int(time.time())}.faiss"
            temp_path = os.path.join(temp_dir, temp_filename)
            
            try:
                # 保存到临时文件（使用英文路径）
                faiss.write_index(self.faiss_index, temp_path)
                
                # 移动到目标位置
                target_path = 'models/ecapa_similarity_database.faiss'
                if os.path.exists(target_path):
                    os.remove(target_path)
                shutil.move(temp_path, target_path)
                
                self.logger.info(f"FAISS索引已保存: {target_path}")
                
            except Exception as e:
                # 清理临时文件
                if os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except:
                        pass
                raise e
            
        except Exception as e:
            self.logger.warning(f"FAISS索引保存失败: {e}")
    
    @handle_exceptions(default_return=[], log_error=True)
    def search_similar_vehicles(self, 
                               audio_path: str, 
                               top_k: Optional[int] = None) -> List[SearchResult]:
        """
        搜索相似车辆
        
        Args:
            audio_path: 查询音频路径
            top_k: 返回结果数量
            
        Returns:
            List[SearchResult]: 搜索结果列表
        """
        start_time = time.time()
        
        try:
            if not self.vehicle_profiles:
                self.logger.warning("车辆档案数据库为空")
                return []
            
            # 使用公共工具提取查询特征
            query_features = extract_and_validate_features(self.feature_extractor, audio_path)
            
            # 使用公共工具标准化特征
            processor = FeatureProcessor()
            query_features = processor.normalize_features(query_features)
            
            # 确定搜索数量
            k = min(top_k or self.max_candidates, len(self.vehicle_profiles))
            
            # 执行搜索
            if FAISS_AVAILABLE and self.faiss_index is not None:
                results = self._faiss_search(query_features, k)
            else:
                results = self._sklearn_search(query_features, k)
            
            # 过滤低相似度结果
            filtered_results = [
                r for r in results 
                if r.similarity_score >= self.similarity_threshold
            ]
            
            # 更新统计
            search_time = time.time() - start_time
            self.stats_manager.update_search_stats(search_time, len(filtered_results) > 0)
            
            self.logger.info(f"搜索完成，找到 {len(filtered_results)} 个匹配结果，耗时: {search_time:.3f}s")
            
            return filtered_results
            
        except Exception as e:
            self.logger.error(f"相似度搜索失败: {e}")
            return []
    
    def _faiss_search(self, query_features: np.ndarray, k: int) -> List[SearchResult]:
        """使用FAISS进行高性能搜索"""
        try:
            query = query_features.reshape(1, -1).astype(np.float32)
            
            # 执行搜索 - L2距离搜索，获取更多候选结果以便分组
            search_k = min(len(self.vehicle_ids), k * 10)  # 获取更多候选结果
            distances, indices = self.faiss_index.search(query, search_k)
            
            # 构建相似度数组和车辆ID列表用于分组
            similarities = []
            candidate_vehicle_ids = []
            
            for distance, idx in zip(distances[0], indices[0]):
                if idx == -1:  # FAISS返回-1表示无效结果
                    continue
                
                if idx < len(self.vehicle_ids):
                    vehicle_id = self.vehicle_ids[idx]
                    # 将L2距离转换为余弦相似度 (0-1范围)
                    similarity_score = 1.0 - (float(distance)**2 / 2.0)
                    similarities.append(similarity_score)
                    candidate_vehicle_ids.append(vehicle_id)
            
            if not similarities:
                return []
            
            # 使用修改后的分组逻辑，按车辆类型分组
            similarity_results = SimilarityCalculator.search_without_threshold(
                similarities=np.array(similarities),
                vehicle_ids=candidate_vehicle_ids,
                vehicle_profiles=self.vehicle_profiles,
                top_k=k
            )
            
            # 转换为SearchResult格式
            results = []
            for rank, (vehicle_id, similarity, metadata) in enumerate(similarity_results):
                result = SearchResult(
                    vehicle_id=vehicle_id,
                    similarity_score=similarity,
                    metadata=metadata,
                    search_time=0.0,  # 由外层函数计算
                    rank=rank + 1
                )
                results.append(result)
            
            return results
            
        except Exception as e:
            self.logger.error(f"FAISS搜索失败: {e}")
            return self._sklearn_search(query_features, k)
    
    def _sklearn_search(self, query_features: np.ndarray, k: int) -> List[SearchResult]:
        """使用scikit-learn进行基础搜索（备用方案）"""
        try:
            # 使用公共工具构建特征矩阵
            processor = FeatureProcessor()
            
            # 构建特征矩阵
            feature_matrix, vehicle_list = processor.build_feature_matrix(self.vehicle_profiles)
            
            if feature_matrix is None or len(vehicle_list) == 0:
                return []
            
            # 计算相似度
            similarities = SimilarityCalculator.compute_cosine_similarity(
                query_features, feature_matrix
            )
            
            # 使用修改后的过滤逻辑，按车辆类型分组
            similarity_results = SimilarityCalculator.search_without_threshold(
                similarities=similarities,
                vehicle_ids=vehicle_list,
                vehicle_profiles=self.vehicle_profiles,
                top_k=k
            )
            
            # 转换为SearchResult格式
            results = []
            for rank, (vehicle_id, similarity, metadata) in enumerate(similarity_results):
                result = SearchResult(
                    vehicle_id=vehicle_id,
                    similarity_score=similarity,
                    metadata=metadata,
                    search_time=0.0,
                    rank=rank + 1
                )
                results.append(result)
            
            return results
            
        except Exception as e:
            self.logger.error(f"基础搜索失败: {e}")
            return []
    
    def save_database(self, save_path: str) -> bool:
        """保存数据库"""
        try:
            with self._lock:
                # 准备保存数据
                save_data = {
                    'profiles': {},
                    'vehicle_ids': self.vehicle_ids,
                    'stats': self.stats_manager.stats,
                    'config': {
                        'feature_dim': self.feature_dim,
                        'similarity_threshold': self.similarity_threshold,
                        'max_candidates': self.max_candidates,
                        'index_type': self.index_type
                    },
                    'version': '2.0'
                }
                
                # 序列化车辆档案
                for vehicle_id, profile in self.vehicle_profiles.items():
                    save_data['profiles'][vehicle_id] = {
                        'vehicle_id': profile.vehicle_id,
                        'features': profile.features.tolist(),
                        'metadata': profile.metadata,
                        'created_at': profile.created_at,
                        'feature_hash': getattr(profile, 'feature_hash', None)
                    }
                
                # 使用公共工具模块保存文件
                file_manager.ensure_dir(os.path.dirname(save_path))
                with open(save_path, 'w', encoding='utf-8') as f:
                    json.dump(save_data, f, indent=2, ensure_ascii=False)
                
                # 保存FAISS索引
                if FAISS_AVAILABLE and self.faiss_index is not None:
                    # 使用系统临时目录避免中文字符问题
                    import tempfile
                    import shutil
                    
                    temp_dir = tempfile.gettempdir()  # 获取系统临时目录
                    temp_filename = f"faiss_index_{int(time.time())}.faiss"
                    temp_path = os.path.join(temp_dir, temp_filename)
                    
                    try:
                        # 保存到临时文件（使用英文路径）
                        faiss.write_index(self.faiss_index, temp_path)
                        
                        # 移动到目标位置
                        target_path = "models/ecapa_similarity_database.faiss"
                        if os.path.exists(target_path):
                            os.remove(target_path)
                        shutil.move(temp_path, target_path)
                        
                        self.logger.info(f"FAISS索引已保存: {target_path}")
                    except Exception as e:
                        # 清理临时文件
                        if os.path.exists(temp_path):
                            try:
                                os.remove(temp_path)
                            except:
                                pass
                        self.logger.warning(f"FAISS索引保存失败: {e}")
                
                self.logger.info(f"数据库保存成功: {save_path}")
                return True
                
        except Exception as e:
            self.logger.error(f"数据库保存失败: {e}")
            return False
    
    def load_database(self, load_path: str) -> bool:
        """加载数据库"""
        try:
            with self._lock:
                if not file_manager.file_exists(load_path):
                    self.logger.warning(f"数据库文件不存在: {load_path}")
                    return False
                
                # 加载数据
                with open(load_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 清空现有数据
                self.vehicle_profiles.clear()
                self.vehicle_ids.clear()
                
                # 恢复车辆档案
                profiles_data = data.get('profiles', {})
                for vehicle_id, profile_data in profiles_data.items():
                    # 处理时间戳字段的兼容性
                    created_at_value = profile_data.get('created_at')
                    if created_at_value is None:
                        # 兼容旧格式的timestamp字段
                        created_at_value = profile_data.get('timestamp', time.time())
                    
                    profile = VehicleProfile(
                        vehicle_id=profile_data['vehicle_id'],
                        features=np.array(profile_data['features']),
                        metadata=profile_data.get('metadata', {}),
                        created_at=created_at_value
                    )
                    self.vehicle_profiles[vehicle_id] = profile
                
                # 恢复其他数据
                self.vehicle_ids = list(self.vehicle_profiles.keys())
                saved_stats = data.get('stats', {})
                self.stats_manager.stats.update(saved_stats)
                
                # 尝试加载FAISS索引，如果失败则重建
                # 使用临时文件避免中文字符问题
                index_path = "models/ecapa_similarity_database.faiss"
                index_loaded = False
                
                if FAISS_AVAILABLE and file_manager.file_exists(index_path):
                    try:
                        # 使用系统临时目录避免中文字符问题
                        import tempfile
                        import shutil
                        
                        # 使用系统临时目录创建临时文件
                        temp_dir = tempfile.gettempdir()
                        temp_filename = f"faiss_load_{int(time.time())}.faiss"
                        temp_path = os.path.join(temp_dir, temp_filename)
                        
                        # 复制到临时文件
                        shutil.copy2(index_path, temp_path)
                        
                        try:
                            self.faiss_index = faiss.read_index(temp_path)
                            if self.use_gpu:
                                res = faiss.StandardGpuResources()
                                self.faiss_index = faiss.index_cpu_to_gpu(res, 0, self.faiss_index)
                            
                            # 验证索引是否与当前数据匹配
                            if self.faiss_index.ntotal == len(self.vehicle_profiles):
                                self.logger.debug(f"FAISS索引加载成功: {index_path}")
                                index_loaded = True
                            else:
                                self.logger.debug(f"FAISS索引大小不匹配，将重建")
                                index_loaded = False
                        finally:
                            # 清理临时文件
                            if os.path.exists(temp_path):
                                try:
                                    os.remove(temp_path)
                                except:
                                    pass
                                
                    except Exception as e:
                        self.logger.warning(f"FAISS索引加载失败，将重建: {e}")
                
                # 如果索引未成功加载，则重建
                if not index_loaded:
                    self._rebuild_faiss_index()
                
                self.logger.info(f"数据库加载成功: {load_path}, 车辆档案数: {len(self.vehicle_profiles)}")
                return True
                
        except Exception as e:
            self.logger.error(f"数据库加载失败: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """获取性能统计"""
        stats = self.stats_manager.stats.copy()
        stats.update({
            'faiss_available': FAISS_AVAILABLE,
            'index_type': self.index_type,
            'use_gpu': self.use_gpu,
            'feature_dim': self.feature_dim,
            'similarity_threshold': self.similarity_threshold
        })
        return stats
    
    def optimize_index(self):
        """优化索引性能"""
        if not FAISS_AVAILABLE or self.faiss_index is None:
            return
        
        try:
            # 根据数据规模调整索引参数
            n_profiles = len(self.vehicle_profiles)
            
            if self.index_type == 'ivf':
                # 调整探测数量
                optimal_nprobe = min(50, max(1, n_profiles // 100))
                self.faiss_index.nprobe = optimal_nprobe
                
            elif self.index_type == 'hnsw':
                # 调整搜索参数
                optimal_ef = min(100, max(16, n_profiles // 50))
                self.faiss_index.hnsw.efSearch = optimal_ef
            
            self.logger.info("索引参数优化完成")
            
        except Exception as e:
            self.logger.error(f"索引优化失败: {e}")