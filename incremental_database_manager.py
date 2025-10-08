#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增量数据库管理器
支持增量更新车辆声纹数据库
"""

# 使用公共工具模块统一导入
from core.common_utils import (
    os, sys, time, json, Path,
    List, Dict, Optional, Tuple, Any, Union,
    dataclass, asdict, warnings, np,
    config_manager, file_manager, logger_factory, TimerContext
)

import pickle
import hashlib
from datetime import datetime
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import shutil
import uuid

# 项目内部导入
from core.similarity_utils import FeatureProcessor, SimilarityCalculator, SearchStatistics, VehicleProfile
from core.exceptions import handle_exceptions, VehicleRecognitionError
from optimized_feature_extractor import OptimizedFeatureExtractor, create_optimized_extractor
from advanced_similarity_engine import AdvancedSimilarityEngine

# 初始化日志器
incremental_logger = logger_factory.get_logger("增量数据库管理")

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

@dataclass
class DatabaseVersion:
    """数据库版本信息"""
    version: str
    timestamp: float
    total_vehicles: int
    total_samples: int
    checksum: str
    description: str = ""

@dataclass
class ChangeRecord:
    """变更记录"""
    operation: str  # 'add', 'remove', 'update'
    vehicle_id: str
    audio_path: str
    timestamp: float
    metadata: Dict[str, Any]
    success: bool
    error_message: str = ""

class IncrementalDatabaseManager:
    """增量数据库管理器"""
    
    def __init__(
        self,
        database_path: str = None,
        change_log_path: str = None):
        """
        初始化增量数据库管理器
        
        Args:
            database_path: 数据库文件路径
            change_log_path: 变更日志路径
        """
        # 从config获取默认路径
        from config import PATH_CONFIG
        
        self.database_path = Path(database_path) if database_path else Path(PATH_CONFIG['vector_database_path'])
        self.change_log_path = Path(change_log_path) if change_log_path else Path(PATH_CONFIG['models_dir']) / 'database_changes.json'
        
        # 创建必要目录
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.change_log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 使用公共工具模块的日志器
        self.logger = logger_factory.get_logger("增量数据库管理器")
        
        # 初始化相似度引擎
        self.similarity_engine = None
        self.feature_extractor = None
        
        # 变更记录
        self.change_history: List[ChangeRecord] = []
        self.pending_changes: List[ChangeRecord] = []
        
        # 数据库版本信息
        self.current_version: Optional[DatabaseVersion] = None
        
        # 性能统计
        self.stats = {
            'total_additions': 0,
            'total_removals': 0,
            'total_updates': 0,
            'successful_operations': 0,
            'failed_operations': 0,
            'last_operation_time': 0.0,
            'avg_operation_time': 0.0
        }
        
        # 批量处理配置
        self.batch_size = 20  # 增加批量大小，提高处理效率
        
        self.logger.info("增量数据库管理器初始化完成")
    
    def initialize(self, force_reload: bool = False) -> bool:
        """初始化数据库管理器"""
        try:
            # 初始化优化的特征提取器
            if self.feature_extractor is None:
                self.feature_extractor = create_optimized_extractor(
                    use_cache=True,
                    max_workers=4  # 适中的线程数，避免过度占用资源
                )
                self.logger.info("优化特征提取器初始化完成")
            
            # 初始化相似度引擎
            if self.similarity_engine is None or force_reload:
                # 使用优化的高级相似度引擎
                self.similarity_engine = AdvancedSimilarityEngine()
                
                # 加载现有数据库
                if self.database_path.exists():
                    success = self.similarity_engine.load_database(str(self.database_path))
                    if success:
                        self.logger.info(f"已加载现有数据库: {self.database_path}")
                    else:
                        self.logger.warning("数据库加载失败，将创建新数据库")
                else:
                    self.logger.info("数据库文件不存在，将创建新数据库")
            
            # 加载变更历史
            self._load_change_history()
            
            # 更新版本信息
            self._update_version_info()
            
            return True
            
        except Exception as e:
            self.logger.error(f"初始化失败: {e}")
            return False
    
    @handle_exceptions(default_return=False, log_error=True)
    def add_vehicle_data(self, 
                        vehicle_id: str, 
                        audio_paths: List[str], 
                        metadata: Optional[Dict[str, Any]] = None,
                        batch_size: int = 10,
                        show_progress: bool = True) -> Tuple[bool, List[str]]:
        """
        增量添加车辆数据（优化版本，支持批量特征提取）
        
        Args:
            vehicle_id: 车辆ID
            audio_paths: 音频文件路径列表
            metadata: 元数据
            batch_size: 批处理大小
            show_progress: 是否显示进度条
            
        Returns:
            (成功标志, 失败的文件列表)
        """
        start_time = time.time()
        failed_files = []
        successful_count = 0
        
        self.logger.info(f"开始添加车辆数据: {vehicle_id}, 文件数量: {len(audio_paths)}")
        
        # 过滤已存在的文件
        valid_paths = []
        skipped_count = 0
        for audio_path in audio_paths:
            if not Path(audio_path).exists():
                self.logger.warning(f"音频文件不存在: {audio_path}")
                failed_files.append(audio_path)
                continue
            
            sample_id = self._generate_sample_id(vehicle_id, audio_path)
            if self._sample_exists(sample_id):
                skipped_count += 1
                continue
            
            valid_paths.append(audio_path)
        
        if skipped_count > 0:
            self.logger.info(f"跳过已存在的样本: {skipped_count} 个")
        
        if not valid_paths:
            self.logger.warning("没有有效的音频文件需要处理")
            return len(failed_files) == 0, failed_files

        # 使用优化的特征提取器进行批量处理
        if isinstance(self.feature_extractor, OptimizedFeatureExtractor):
            # 只在处理大批量文件时记录开始日志
            if len(valid_paths) > 10:
                self.logger.info(f"开始批量处理 {len(valid_paths)} 个文件")
            else:
                self.logger.debug(f"开始批量处理 {len(valid_paths)} 个文件")
            
            # 进度条回调函数
            def progress_callback(completed, total, current_file):
                if show_progress:
                    progress = completed / total * 100
                    bar_length = 30
                    filled_length = int(bar_length * completed // total)
                    bar = '█' * filled_length + '░' * (bar_length - filled_length)
                    print(f"\r特征提取进度: [{bar}] {progress:.1f}% ({completed}/{total})", end='', flush=True)
                    if completed == total:
                        print()  # 换行
            
            # 分批处理文件，避免一次性处理过多文件导致系统卡住
            all_extraction_results = []
            total_successful = 0
            total_cache_hits = 0
            total_time = 0.0
            
            total_batches = (len(valid_paths) + self.batch_size - 1) // self.batch_size
            for i in range(0, len(valid_paths), self.batch_size):
                batch_paths = valid_paths[i:i+self.batch_size]
                batch_num = i // self.batch_size + 1
                
                # 批量提取特征
                extraction_results, extraction_stats = self.feature_extractor.extract_batch(
                    batch_paths, 
                    progress_callback=progress_callback
                )
                
                if extraction_results:
                    all_extraction_results.extend(extraction_results)
                    total_successful += extraction_stats.successful_files
                    total_cache_hits += extraction_stats.cache_hits
                    total_time += extraction_stats.total_time
            
            # 创建汇总统计
            from optimized_feature_extractor import BatchExtractionStats
            extraction_stats = BatchExtractionStats(
                total_files=len(valid_paths),
                successful_files=total_successful,
                failed_files=len(valid_paths) - total_successful,
                cache_hits=total_cache_hits,
                cache_misses=len(valid_paths) - total_cache_hits,
                total_time=total_time,
                avg_time_per_file=total_time / len(valid_paths) if valid_paths else 0
            )
            
            # 只在处理大批量文件或有失败时记录详细日志
            if extraction_stats.total_files > 10 or extraction_stats.failed_files > 0:
                self.logger.info(
                    f"特征提取完成: {extraction_stats.successful_files}/{extraction_stats.total_files} 成功 "
                    f"(耗时: {extraction_stats.total_time:.2f}s, 缓存命中: {extraction_stats.cache_hits})"
                )
            else:
                # 小批量文件使用debug级别日志
                self.logger.debug(
                    f"特征提取完成: {extraction_stats.successful_files}/{extraction_stats.total_files} 成功 "
                    f"(耗时: {extraction_stats.total_time:.2f}s, 缓存命中: {extraction_stats.cache_hits})"
                )
            
            # 处理提取结果 - 使用进度条显示
            if show_progress and all_extraction_results:
                print("正在保存到数据库...")
            
            for i, result in enumerate(all_extraction_results):
                try:
                    # 显示保存进度条
                    if show_progress:
                        progress = (i + 1) / len(all_extraction_results) * 100
                        bar_length = 30
                        filled_length = int(bar_length * (i + 1) // len(all_extraction_results))
                        bar = '█' * filled_length + '░' * (bar_length - filled_length)
                        print(f"\r保存进度: [{bar}] {progress:.1f}% ({i + 1}/{len(all_extraction_results)})", end='', flush=True)
                        if i + 1 == len(all_extraction_results):
                            print()  # 换行
                    
                    if not result.success:
                        failed_files.append(result.file_path)
                        continue
                    
                    # 生成样本ID
                    sample_id = self._generate_sample_id(vehicle_id, result.file_path)
                    
                    # 准备元数据
                    sample_metadata = metadata.copy() if metadata else {}
                    sample_metadata.update({
                        'vehicle_id': vehicle_id,
                        'audio_path': result.file_path,
                        'sample_id': sample_id,
                        'added_time': time.time(),
                        'extraction_time': result.extraction_time,
                        'audio_duration': result.audio_duration,
                        'file_size': result.file_size
                    })
                    
                    # 直接添加到相似度引擎（跳过重复的特征提取）
                    # 提取特征数据（处理字典格式）
                    if isinstance(result.features, dict):
                        # 特征提取器返回字典格式，提取实际特征
                        if 'fused_features' in result.features:
                            features_array = result.features['fused_features']
                        else:
                            self.logger.error(f"特征字典中缺少'fused_features'键: {sample_id}")
                            failed_files.append(result.file_path)
                            continue
                    else:
                        # 直接是numpy数组
                        features_array = result.features
                    
                    # 使用公共工具标准化特征
                    from core.similarity_utils import FeatureProcessor
                    processor = FeatureProcessor()
                    normalized_features = processor.normalize_features(features_array)
                    
                    # 创建VehicleProfile对象
                    profile = VehicleProfile(
                        vehicle_id=sample_id,
                        features=normalized_features,
                        metadata=sample_metadata,
                        created_at=time.time()
                    )
                    
                    # 直接添加到数据库
                    if hasattr(self.similarity_engine, 'vehicle_profiles'):
                        self.similarity_engine.vehicle_profiles[sample_id] = profile
                        successful_count += 1
                        
                        # 添加到FAISS索引
                        if hasattr(self.similarity_engine, '_add_to_faiss_index'):
                            self.similarity_engine._add_to_faiss_index(profile)
                        
                        # 记录成功的变更
                        change_record = ChangeRecord(
                            operation='add',
                            vehicle_id=vehicle_id,
                            audio_path=result.file_path,
                            timestamp=time.time(),
                            metadata=sample_metadata,
                            success=True
                        )
                        self.change_history.append(change_record)
                        
                        # 只在重要节点显示成功信息
                        if i % 20 == 0 or i == len(all_extraction_results) - 1:
                            self.logger.info(f"成功添加样本: {sample_id}")
                    else:
                        # 回退到标准方法，传递已提取的特征
                        success = self.similarity_engine.add_vehicle_profile(
                            vehicle_id=sample_id,
                            audio_path=result.file_path,
                            features=normalized_features,
                            metadata=sample_metadata
                        )
                        
                        if success:
                            successful_count += 1
                            change_record = ChangeRecord(
                                operation='add',
                                vehicle_id=vehicle_id,
                                audio_path=result.file_path,
                                timestamp=time.time(),
                                metadata=sample_metadata,
                                success=True
                            )
                            self.change_history.append(change_record)
                            
                            # 只在重要节点显示成功信息
                            if i % 20 == 0 or i == len(all_extraction_results) - 1:
                                self.logger.info(f"成功添加样本(回退方法): {sample_id}")
                        else:
                            failed_files.append(result.file_path)
                            self.logger.warning(f"添加样本失败(回退方法): {result.file_path}")
                            
                except Exception as e:
                    failed_files.append(result.file_path)
                    self.logger.error(f"处理提取结果失败 {result.file_path}: {e}")
                    
                    # 记录失败的变更
                    change_record = ChangeRecord(
                        operation='add',
                        vehicle_id=vehicle_id,
                        audio_path=result.file_path,
                        timestamp=time.time(),
                        metadata=metadata or {},
                        success=False,
                        error_message=str(e)
                    )
                    self.change_history.append(change_record)
        
        # 保存数据库
        if successful_count > 0:
            self._save_database()
            self._save_change_history()
            self._update_version_info()
        
        # 更新统计
        operation_time = time.time() - start_time
        self._update_stats('add', successful_count, len(failed_files), operation_time)
        
        success = len(failed_files) == 0
        self.logger.info(f"车辆数据添加完成: {vehicle_id}, "
                        f"成功: {successful_count}, 失败: {len(failed_files)}, "
                        f"耗时: {operation_time:.2f}s")
        
        return success, failed_files
    
    @handle_exceptions(default_return=False, log_error=True)
    def remove_vehicle_data(self, vehicle_id: str, sample_ids: Optional[List[str]] = None) -> bool:
        """
        删除车辆数据
        
        Args:
            vehicle_id: 车辆ID
            sample_ids: 要删除的样本ID列表，如果为None则删除该车辆的所有样本
            
        Returns:
            成功标志
        """
        start_time = time.time()
        
        self.logger.info(f"开始删除车辆数据: {vehicle_id}")
        
        try:
            # 获取要删除的样本
            if sample_ids is None:
                # 删除该车辆的所有样本
                sample_ids = self._get_vehicle_samples(vehicle_id)
            
            removed_count = 0
            for sample_id in sample_ids:
                success = self.similarity_engine.remove_vehicle_profile(sample_id)
                if success:
                    removed_count += 1
                    
                    # 记录变更
                    change_record = ChangeRecord(
                        operation='remove',
                        vehicle_id=vehicle_id,
                        audio_path=sample_id,
                        timestamp=time.time(),
                        metadata={'sample_id': sample_id},
                        success=True
                    )
                    self.change_history.append(change_record)
                    
                    self.logger.debug(f"成功删除样本: {sample_id}")
                else:
                    self.logger.warning(f"删除样本失败: {sample_id}")
            
            # 保存数据库
            if removed_count > 0:
                self._save_database()
                self._save_change_history()
                self._update_version_info()
            
            # 更新统计
            operation_time = time.time() - start_time
            self._update_stats('remove', removed_count, len(sample_ids) - removed_count, operation_time)
            
            self.logger.info(f"车辆数据删除完成: {vehicle_id}, "
                           f"删除数量: {removed_count}/{len(sample_ids)}, "
                           f"耗时: {operation_time:.2f}s")
            
            return removed_count > 0
            
        except Exception as e:
            self.logger.error(f"删除车辆数据失败: {e}")
            return False
    
    @handle_exceptions(default_return=False, log_error=True)
    def update_vehicle_metadata(self, vehicle_id: str, new_metadata: Dict[str, Any]) -> bool:
        """
        更新车辆元数据
        
        Args:
            vehicle_id: 车辆ID
            new_metadata: 新的元数据
            
        Returns:
            成功标志
        """
        start_time = time.time()
        
        self.logger.info(f"开始更新车辆元数据: {vehicle_id}")
        
        try:
            # 获取该车辆的所有样本
            sample_ids = self._get_vehicle_samples(vehicle_id)
            
            if not sample_ids:
                self.logger.warning(f"未找到车辆样本: {vehicle_id}")
                return False
            
            updated_count = 0
            for sample_id in sample_ids:
                # 获取现有档案
                profile = self.similarity_engine.vehicle_profiles.get(sample_id)
                if profile:
                    # 更新元数据
                    profile.metadata.update(new_metadata)
                    profile.metadata['updated_time'] = time.time()
                    updated_count += 1
                    
                    self.logger.debug(f"成功更新样本元数据: {sample_id}")
            
            # 保存数据库
            if updated_count > 0:
                self._save_database()
                
                # 记录变更
                change_record = ChangeRecord(
                    operation='update',
                    vehicle_id=vehicle_id,
                    audio_path='metadata_update',
                    timestamp=time.time(),
                    metadata=new_metadata,
                    success=True
                )
                self.change_history.append(change_record)
                self._save_change_history()
                self._update_version_info()
            
            # 更新统计
            operation_time = time.time() - start_time
            self._update_stats('update', updated_count, 0, operation_time)
            
            self.logger.info(f"车辆元数据更新完成: {vehicle_id}, "
                           f"更新数量: {updated_count}, "
                           f"耗时: {operation_time:.2f}s")
            
            return updated_count > 0
            
        except Exception as e:
            self.logger.error(f"更新车辆元数据失败: {e}")
            return False
    
    def batch_add_from_directory(self, 
                                directory: str, 
                                vehicle_mapping: Optional[Dict[str, str]] = None,
                                file_extensions: List[str] = ['.wav', '.mp3', '.flac']) -> Dict[str, Any]:
        """
        从目录批量添加车辆数据
        
        Args:
            directory: 音频文件目录
            vehicle_mapping: 文件夹名到车辆ID的映射
            file_extensions: 支持的文件扩展名
            
        Returns:
            处理结果统计
        """
        directory = Path(directory)
        if not directory.exists():
            self.logger.error(f"目录不存在: {directory}")
            return {}
        
        results = {
            'total_vehicles': 0,
            'successful_vehicles': 0,
            'failed_vehicles': 0,
            'total_files': 0,
            'successful_files': 0,
            'failed_files': 0,
            'processing_time': 0.0,
            'details': {}
        }
        
        start_time = time.time()
        
        # 收集所有有效的车辆目录
        valid_vehicle_dirs = []
        for vehicle_dir in directory.iterdir():
            if not vehicle_dir.is_dir():
                continue
            
            # 收集音频文件
            audio_files = []
            for ext in file_extensions:
                audio_files.extend(vehicle_dir.glob(f"*{ext}"))
                audio_files.extend(vehicle_dir.glob(f"*{ext.upper()}"))
            
            if audio_files:
                valid_vehicle_dirs.append((vehicle_dir, audio_files))
        
        if not valid_vehicle_dirs:
            self.logger.warning("未找到包含音频文件的车辆目录")
            return results
        
        total_batches = len(valid_vehicle_dirs)
        # 只记录一次批量处理开始的日志，避免重复
        self.logger.info(f"开始批量处理 {total_batches} 个车辆目录")
        
        # 遍历车辆目录，显示批次级别进度条
        for batch_idx, (vehicle_dir, audio_files) in enumerate(valid_vehicle_dirs, 1):
            # 确定车辆ID
            vehicle_id = vehicle_mapping.get(vehicle_dir.name, vehicle_dir.name) if vehicle_mapping else vehicle_dir.name
            
            results['total_vehicles'] += 1
            results['total_files'] += len(audio_files)
            
            # 显示批次进度条
            progress = batch_idx / total_batches * 100
            bar_length = 40
            filled_length = int(bar_length * batch_idx // total_batches)
            bar = '█' * filled_length + '░' * (bar_length - filled_length)
            print(f"\r总进度: [{bar}] {progress:.1f}% ({batch_idx}/{total_batches}) - 处理: {vehicle_id}", end='', flush=True)
            
            # 添加车辆数据（关闭单个批次的进度条显示）
            success, failed_files = self.add_vehicle_data(
                vehicle_id=vehicle_id,
                audio_paths=[str(f) for f in audio_files],
                metadata={'source_directory': str(vehicle_dir)},
                show_progress=False  # 关闭文件级别的进度条
            )
            
            # 统计结果
            successful_files = len(audio_files) - len(failed_files)
            results['successful_files'] += successful_files
            results['failed_files'] += len(failed_files)
            
            if success:
                results['successful_vehicles'] += 1
            else:
                results['failed_vehicles'] += 1
            
            results['details'][vehicle_id] = {
                'total_files': len(audio_files),
                'successful_files': successful_files,
                'failed_files': len(failed_files),
                'success': success
            }
            
            # 完成当前批次后换行并显示结果
            print(f" - 完成: {successful_files}/{len(audio_files)} 文件")
        
        # 最终进度条显示
        print(f"\r总进度: [{'█' * bar_length}] 100.0% ({total_batches}/{total_batches}) - 全部完成")
        
        results['processing_time'] = time.time() - start_time
        
        self.logger.info(f"批量添加完成: "
                        f"车辆: {results['successful_vehicles']}/{results['total_vehicles']}, "
                        f"文件: {results['successful_files']}/{results['total_files']}, "
                        f"耗时: {results['processing_time']:.2f}s")
        
        return results
    
    def _generate_sample_id(self, vehicle_id: str, audio_path: str) -> str:
        """生成唯一的样本ID"""
        # 使用车辆ID和文件路径的哈希生成唯一ID
        path_hash = hashlib.md5(audio_path.encode()).hexdigest()[:8]
        filename = Path(audio_path).stem
        return f"{vehicle_id}_{filename}_{path_hash}"
    
    def _sample_exists(self, sample_id: str) -> bool:
        """检查样本是否已存在"""
        return sample_id in self.similarity_engine.vehicle_profiles
    
    def _get_vehicle_samples(self, vehicle_id: str) -> List[str]:
        """获取指定车辆的所有样本ID"""
        samples = []
        for sample_id, profile in self.similarity_engine.vehicle_profiles.items():
            if profile.metadata.get('vehicle_id') == vehicle_id:
                samples.append(sample_id)
        return samples
    
    def _create_backup(self, description: str = "") -> str:
        """创建数据库备份 - 已禁用"""
        self.logger.warning("备份功能已被禁用")
        return ""

    def _cleanup_old_backups(self):
        """清理旧备份文件 - 已禁用"""
        pass

    def _save_database(self) -> bool:
        """保存数据库"""
        try:
            return self.similarity_engine.save_database(str(self.database_path))
        except Exception as e:
            self.logger.error(f"保存数据库失败: {e}")
            return False
    
    def _load_change_history(self):
        """加载变更历史"""
        try:
            if self.change_log_path.exists():
                with open(self.change_log_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                self.change_history = []
                for record_data in data.get('changes', []):
                    record = ChangeRecord(**record_data)
                    self.change_history.append(record)
                
                self.stats = data.get('stats', self.stats)
                self.logger.info(f"加载变更历史: {len(self.change_history)} 条记录")
                
        except Exception as e:
            self.logger.error(f"加载变更历史失败: {e}")
    
    def _save_change_history(self):
        """保存变更历史"""
        try:
            data = {
                'changes': [asdict(record) for record in self.change_history],
                'stats': self.stats,
                'last_updated': time.time()
            }
            
            with open(self.change_log_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            self.logger.error(f"保存变更历史失败: {e}")
    
    def _update_version_info(self):
        """更新版本信息"""
        try:
            total_vehicles = len(set(
                profile.metadata.get('vehicle_id', '') 
                for profile in self.similarity_engine.vehicle_profiles.values()
            ))
            total_samples = len(self.similarity_engine.vehicle_profiles)
            
            # 计算数据库校验和
            checksum = self._calculate_database_checksum()
            
            self.current_version = DatabaseVersion(
                version=datetime.now().strftime("%Y%m%d_%H%M%S"),
                timestamp=time.time(),
                total_vehicles=total_vehicles,
                total_samples=total_samples,
                checksum=checksum,
                description=f"数据库版本 - {total_vehicles}辆车, {total_samples}个样本"
            )
            
        except Exception as e:
            self.logger.error(f"更新版本信息失败: {e}")
    
    def _calculate_database_checksum(self) -> str:
        """计算数据库校验和"""
        try:
            if self.database_path.exists():
                with open(self.database_path, 'rb') as f:
                    return hashlib.md5(f.read()).hexdigest()
        except Exception as e:
            self.logger.error(f"计算校验和失败: {e}")
        return ""
    
    def _update_stats(self, operation: str, success_count: int, fail_count: int, operation_time: float):
        """更新统计信息"""
        if operation == 'add':
            self.stats['total_additions'] += success_count
        elif operation == 'remove':
            self.stats['total_removals'] += success_count
        elif operation == 'update':
            self.stats['total_updates'] += success_count
        
        self.stats['successful_operations'] += success_count
        self.stats['failed_operations'] += fail_count
        self.stats['last_operation_time'] = operation_time
        
        # 更新平均操作时间
        total_ops = self.stats['successful_operations'] + self.stats['failed_operations']
        if total_ops > 0:
            total_time = self.stats['avg_operation_time'] * (total_ops - success_count - fail_count)
            self.stats['avg_operation_time'] = (total_time + operation_time) / total_ops
    
    def get_database_info(self) -> Dict[str, Any]:
        """获取数据库信息"""
        if not self.similarity_engine:
            return {}
        
        # 统计车辆信息
        vehicle_stats = {}
        
        # 检查是否使用高级引擎
        if hasattr(self.similarity_engine, 'advanced_engine') and self.similarity_engine.advanced_engine:
            # 使用高级引擎的数据
            profiles = self.similarity_engine.advanced_engine.vehicle_profiles
        else:
            # 使用基础引擎的数据
            profiles = self.similarity_engine.vehicle_profiles
        
        for profile in profiles.values():
            # 从metadata中获取vehicle_id，如果没有则从profile的vehicle_id中提取
            if hasattr(profile, 'metadata') and profile.metadata:
                vehicle_id = profile.metadata.get('vehicle_id', 'unknown')
            else:
                # 从sample_id中提取车辆ID（格式：vehicle_type_number_...）
                sample_id = profile.vehicle_id
                parts = sample_id.split('_')
                if len(parts) >= 2:
                    vehicle_id = f"{parts[0]}_{parts[1]}"
                else:
                    vehicle_id = 'unknown'
            
            if vehicle_id not in vehicle_stats:
                vehicle_stats[vehicle_id] = 0
            vehicle_stats[vehicle_id] += 1
        
        # 计算数据库文件大小
        database_size_mb = 0
        if self.database_path.exists():
            database_size_mb = self.database_path.stat().st_size / (1024 * 1024)
        
        # 计算索引大小（如果存在）
        index_size_mb = 0
        index_path = self.database_path.parent / "faiss_index.bin"
        if index_path.exists():
            index_size_mb = index_path.stat().st_size / (1024 * 1024)
        
        # 获取最后更新时间
        last_updated = "未知"
        if self.database_path.exists():
            last_updated = datetime.fromtimestamp(self.database_path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        
        # 获取特征维度
        feature_dimension = 0
        if profiles:
            first_profile = next(iter(profiles.values()))
            if hasattr(first_profile, 'features') and first_profile.features is not None:
                feature_dimension = len(first_profile.features)
        
        return {
            'database_path': str(self.database_path),
            'total_vehicles': len(vehicle_stats),
            'total_files': len(profiles),  # 添加缺失的total_files字段
            'total_samples': len(profiles),
            'feature_dimension': feature_dimension,
            'database_size_mb': database_size_mb,
            'index_size_mb': index_size_mb,
            'last_updated': last_updated,
            'vehicle_distribution': vehicle_stats,
            'current_version': asdict(self.current_version) if self.current_version else None,
            'stats': self.stats,
            'recent_changes': len([r for r in self.change_history if time.time() - r.timestamp < 86400])  # 24小时内的变更
        }
    
    def get_recent_changes(self, hours: int = 24) -> List[ChangeRecord]:
        """获取最近的变更记录"""
        cutoff_time = time.time() - (hours * 3600)
        return [record for record in self.change_history if record.timestamp >= cutoff_time]
    
    def restore_from_backup(self, backup_path: str) -> bool:
        """从备份恢复数据库 - 已禁用"""
        self.logger.error("备份恢复功能已被禁用")
        return False

    def list_backups(self) -> List[Dict[str, Any]]:
        """列出所有备份文件 - 已禁用"""
        self.logger.warning("备份列表功能已被禁用")
        return []
    
    def optimize_database(self) -> bool:
        """优化数据库（重建索引等）"""
        try:
            self.logger.info("开始优化数据库...")
            
            # 重建特征矩阵
            if hasattr(self.similarity_engine, '_rebuild_feature_matrix'):
                self.similarity_engine._rebuild_feature_matrix()
            
            # 保存优化后的数据库
            self._save_database()
            self._update_version_info()
            
            self.logger.info("数据库优化完成")
            return True
            
        except Exception as e:
            self.logger.error(f"数据库优化失败: {e}")
            return False
    
    def set_batch_size(self, batch_size: int) -> None:
        """设置批量处理大小
        
        Args:
            batch_size: 批量处理大小，较大的值可提高处理效率
        """
        if batch_size < 1:
            batch_size = 1
        elif batch_size > 50:  # 增加最大批次限制
            batch_size = 50
            
        self.batch_size = batch_size
        self.logger.info(f"设置批量处理大小为: {batch_size}")
        
        # 如果特征提取器支持设置批量大小，也进行设置
        if hasattr(self.feature_extractor, 'set_batch_size'):
            self.feature_extractor.set_batch_size(batch_size)

# 使用示例和工具函数
def create_incremental_manager(database_path: str = None) -> IncrementalDatabaseManager:
    """创建增量数据库管理器实例"""
    if database_path is None:
        from config import PATH_CONFIG
        database_path = PATH_CONFIG['vector_database_path']
    
    manager = IncrementalDatabaseManager(database_path=database_path)
    manager.initialize()
    return manager

def quick_add_vehicle(manager: IncrementalDatabaseManager, 
                     vehicle_id: str, 
                     audio_directory: str) -> bool:
    """快速添加单个车辆的所有音频文件"""
    audio_dir = Path(audio_directory)
    if not audio_dir.exists():
        print(f"目录不存在: {audio_directory}")
        return False
    
    # 收集音频文件
    audio_files = []
    for ext in ['.wav', '.mp3', '.flac', '.WAV', '.MP3', '.FLAC']:
        audio_files.extend(audio_dir.glob(f"*{ext}"))
    
    if not audio_files:
        print(f"目录中未找到音频文件: {audio_directory}")
        return False
    
    print(f"找到 {len(audio_files)} 个音频文件")
    
    # 添加车辆数据
    success, failed_files = manager.add_vehicle_data(
        vehicle_id=vehicle_id,
        audio_paths=[str(f) for f in audio_files],
        metadata={'source_directory': str(audio_directory)}
    )
    
    if success:
        print(f"成功添加车辆: {vehicle_id}")
    else:
        print(f"添加车辆失败: {vehicle_id}, 失败文件数: {len(failed_files)}")
    
    return success

if __name__ == "__main__":
    # 示例用法
    print("增量数据库管理器示例")
    
    # 创建管理器
    manager = create_incremental_manager()
    
    # 显示数据库信息
    info = manager.get_database_info()
    print(f"当前数据库: {info['total_vehicles']} 辆车, {info['total_samples']} 个样本")
    
    # 示例：添加新车辆数据
    # success, failed = manager.add_vehicle_data(
    #     vehicle_id="new_car_001",
    #     audio_paths=["path/to/audio1.wav", "path/to/audio2.wav"],
    #     metadata={"type": "sedan", "year": 2023}
    # )
    
    print("管理器初始化完成")