#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
优化的特征提取器
支持批量处理、缓存、多线程和性能优化
"""

# 使用公共工具模块统一导入
from core.common_utils import (
    os, sys, time, json, Path,
    List, Dict, Optional, Tuple, Any, Union,
    dataclass, asdict, warnings, np,
    config_manager, file_manager, logger_factory, TimerContext
)

import torch
import librosa
import hashlib
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from queue import Queue
import gc

# 项目内部导入
from core.similarity_utils import FeatureProcessor
from core.exceptions import handle_exceptions, VehicleRecognitionError

# 初始化日志器
feature_logger = logger_factory.get_logger("优化特征提取器")

import torchaudio

# 忽略一些不重要的警告
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

@dataclass
class ExtractionResult:
    """特征提取结果"""
    file_path: str
    success: bool
    features: Optional[np.ndarray] = None
    error_message: Optional[str] = None
    extraction_time: float = 0.0
    file_size: int = 0
    audio_duration: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = asdict(self)
        if self.features is not None:
            result['features'] = self.features.tolist()
        return result

@dataclass
class BatchExtractionStats:
    """批量提取统计"""
    total_files: int = 0
    successful_files: int = 0
    failed_files: int = 0
    total_time: float = 0.0
    avg_time_per_file: float = 0.0
    total_audio_duration: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)

class FeatureCache:
    """特征缓存管理器"""
    
    def __init__(self, cache_dir: str = "feature_cache", max_cache_size: int = 1000):
        self.cache_dir = Path(cache_dir)
        self.max_cache_size = max_cache_size
        self.logger = logger_factory.get_feature_logger()
        
        # 使用公共文件管理器
        file_manager.ensure_dir(self.cache_dir)
        
        self.cache_index_path = self.cache_dir / "cache_index.json"
        self.cache_index = file_manager.safe_load_json(self.cache_index_path, {})
        
        self._lock = threading.Lock()

    def _get_file_hash(self, file_path: str) -> str:
        """计算文件哈希值"""
        return file_manager.get_file_hash(file_path)

    def _save_cache_index_no_lock(self):
        """保存缓存索引到文件（不获取锁）"""
        try:
            # 确保目录存在
            self.cache_index_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 保存索引文件
            with open(self.cache_index_path, 'w', encoding='utf-8') as f:
                json.dump(self.cache_index, f, indent=2, ensure_ascii=False)
            
            # 验证文件是否保存成功
            if not self.cache_index_path.exists():
                raise Exception("缓存索引文件保存后不存在")
                
        except Exception as e:
            self.logger.error(f"保存缓存索引失败: {e}")
            raise

    def _save_cache_index(self):
        """保存缓存索引"""
        with self._lock:
            self._save_cache_index_no_lock()

    def get(self, file_path: str) -> Optional[np.ndarray]:
        """从缓存获取特征"""
        with self._lock:
            file_hash = self._get_file_hash(file_path)
            
            if not file_hash or file_hash not in self.cache_index:
                return None
            
            cache_info = self.cache_index[file_hash]
            cache_file = self.cache_dir / f"{file_hash}.pkl"
            
            if not cache_file.exists():
                # 缓存文件不存在，清理索引
                del self.cache_index[file_hash]
                self._save_cache_index()
                return None
            
            try:
                with open(cache_file, 'rb') as f:
                    features = pickle.load(f)
                
                # 更新访问时间
                cache_info['last_access'] = time.time()
                cache_info['access_count'] = cache_info.get('access_count', 0) + 1
                # 直接调用无锁版本，因为当前已经持有锁
                self._save_cache_index_no_lock()
                
                return features
            except Exception as e:
                self.logger.warning(f"读取缓存失败: {e}")
                return None
    
    def put(self, file_path: str, features: np.ndarray):
        """将特征存入缓存"""
        with self._lock:
            file_hash = self._get_file_hash(file_path)
            if not file_hash:
                self.logger.warning(f"无法获取文件哈希: {Path(file_path).name}")
                return
            
            # 检查缓存大小限制
            if len(self.cache_index) >= self.max_cache_size:
                self._cleanup_cache()
            
            cache_file = self.cache_dir / f"{file_hash}.pkl"
            
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(features, f)
                
                # 更新索引
                self.cache_index[file_hash] = {
                    'file_path': file_path,
                    'created_time': time.time(),
                    'last_access': time.time(),
                    'access_count': 1,
                    'feature_shape': features.shape if hasattr(features, 'shape') else str(features)
                }
                
                # 直接保存索引，不获取锁（因为已经在锁内）
                self._save_cache_index_no_lock()
                
            except Exception as e:
                self.logger.error(f"保存缓存失败: {e}")
                raise
    
    def _cleanup_cache(self):
        """清理缓存"""
        if len(self.cache_index) < self.max_cache_size:
            return
        
        # 按访问时间排序，删除最旧的缓存
        sorted_items = sorted(
            self.cache_index.items(),
            key=lambda x: x[1].get('last_access', 0)
        )
        
        # 删除最旧的20%
        cleanup_count = max(1, len(sorted_items) // 5)
        
        for file_hash, _ in sorted_items[:cleanup_count]:
            cache_file = self.cache_dir / f"{file_hash}.pkl"
            try:
                if cache_file.exists():
                    cache_file.unlink()
                del self.cache_index[file_hash]
            except Exception as e:
                self.logger.warning(f"清理缓存失败: {e}")
        
        self.logger.info(f"清理了 {cleanup_count} 个缓存文件")
    
    def clear(self):
        """清空所有缓存"""
        with self._lock:
            try:
                for cache_file in self.cache_dir.glob("*.pkl"):
                    cache_file.unlink()
                self.cache_index.clear()
                self._save_cache_index()
                self.logger.info("已清空所有缓存")
            except Exception as e:
                self.logger.error(f"清空缓存失败: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        with self._lock:
            total_size = 0
            for cache_file in self.cache_dir.glob("*.pkl"):
                try:
                    total_size += cache_file.stat().st_size
                except Exception:
                    pass
            
            return {
                'cache_count': len(self.cache_index),
                'cache_size_mb': total_size / (1024 * 1024),
                'max_cache_size': self.max_cache_size,
                'cache_dir': str(self.cache_dir)
            }

class OptimizedFeatureExtractor:
    """优化的特征提取器，集成ECAPA-TDNN功能"""
    
    def __init__(
        self,
        use_cache: bool = True,
        cache_dir: str = "feature_cache",
        max_workers: int = None,
        batch_size: int = 32,
        device: str = "auto",
        target_sr: int = None,
        model_source: str = "speechbrain/spkrec-ecapa-voxceleb",
        model_cache_dir: str = None
    ):
        """
        初始化优化的特征提取器
        
        Args:
            use_cache: 是否使用缓存
            cache_dir: 缓存目录
            max_workers: 最大工作线程数
            batch_size: 批处理大小
            device: 设备类型 ('cpu', 'cuda', 'auto')
            target_sr: 目标采样率
            model_source: 预训练模型源
            model_cache_dir: 模型缓存目录
        """
        # 使用配置管理器获取默认值
        self.target_sr = target_sr or config_manager.get_audio_config('sample_rate', 16000)
        self.batch_size = batch_size
        self.max_workers = max_workers or config_manager.get('ecapa', 'max_workers', 2)
        
        # 使用日志工厂
        self.logger = logger_factory.get_feature_logger()
        
        # 设备配置
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.logger.info(f"使用设备: {self.device}")
        
        # 缓存配置
        self.use_cache = use_cache and config_manager.get('ecapa', 'use_cache', True)
        if self.use_cache:
            self.cache = FeatureCache(cache_dir, max_cache_size=1000)
        else:
            self.cache = None
        
        # 模型配置
        self.model_source = model_source
        if model_cache_dir is None:
            from config import PATH_CONFIG
            self.model_cache_dir = Path(PATH_CONFIG['ecapa_model_path'])
        else:
            self.model_cache_dir = Path(model_cache_dir)
        file_manager.ensure_dir(self.model_cache_dir)
        
        # 初始化模型
        self.model = None
        self.classifier = None
        self.model_loaded = False
        
        # 统计信息
        self.stats = BatchExtractionStats()
        
        # 音频预处理工具
        self.resampler = None
        
        # 初始化ECAPA模型
        try:
            self._load_ecapa_model()
            self.logger.info(f"ECAPA特征提取器初始化成功，设备: {self.device}")
        except Exception as e:
            self.logger.error(f"ECAPA特征提取器初始化失败: {e}")
            raise
        
        self.logger.info(f"优化特征提取器初始化完成，最大工作线程: {self.max_workers}")
    
    def _load_ecapa_model(self):
        """加载ECAPA-TDNN预训练模型"""
        try:
            from speechbrain.pretrained import EncoderClassifier
            
            self.logger.info("正在加载ECAPA-TDNN模型...")
            
            # 创建缓存目录
            file_manager.ensure_dir(self.model_cache_dir)
            
            # 检查核心模型文件
            core_files = [
                "hyperparams.yaml",
                "embedding_model.ckpt",
                "classifier.ckpt", 
                "normalizer.ckpt"
            ]
            
            missing_files = []
            for filename in core_files:
                filepath = self.model_cache_dir / filename
                if not filepath.exists():
                    missing_files.append(filename)
                elif filepath.stat().st_size == 0:
                    self.logger.error(f"模型文件损坏（大小为0）: {filename}")
                    missing_files.append(filename)
            
            # 如果有缺失文件，提供下载指南
            if missing_files:
                self.logger.error(f"ECAPA-TDNN核心模型文件缺失或损坏: {missing_files}")
                self._provide_download_guide(missing_files)
                self.model_loaded = False
                return
            
            # 设置环境变量，强制使用本地文件，禁用所有网络请求
            os.environ["SB_DISABLE_HF_HUB"] = "1"
            os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
            os.environ["HF_HUB_OFFLINE"] = "1"
            
            # 创建一个临时的mean_var_norm_emb.ckpt文件（如果不存在）
            mean_var_file = self.model_cache_dir / "mean_var_norm_emb.ckpt"
            if not mean_var_file.exists():
                self.logger.warning("创建临时的mean_var_norm_emb.ckpt文件")
                self._create_dummy_mean_var_file(mean_var_file)
            
            # 加载预训练模型（使用本地文件）
            self.classifier = EncoderClassifier.from_hparams(
                source=str(self.model_cache_dir),  # 使用本地路径
                run_opts={"device": str(self.device)},
                savedir=str(self.model_cache_dir)
            )
            
            # 设置为评估模式
            self.classifier.eval()
            
            # 验证模型是否正确加载
            if not self._validate_model():
                self.logger.error("模型验证失败")
                self.model_loaded = False
                return
            
            self.model_loaded = True
            self.logger.info("ECAPA-TDNN模型加载成功")
            
        except Exception as e:
            self.logger.error(f"ECAPA-TDNN模型加载失败: {e}")
            self.model_loaded = False
            raise
    
    def _provide_download_guide(self, missing_files):
        """提供模型下载指南"""
        self.logger.info("=" * 60)
        self.logger.info("ECAPA-TDNN模型文件缺失，请按以下步骤下载：")
        self.logger.info("1. 访问 https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb")
        self.logger.info(f"2. 下载以下文件到 {self.model_cache_dir}/ 目录：")
        for file in missing_files:
            self.logger.info(f"   - {file}")
        self.logger.info("3. 或运行以下命令自动下载：")
        self.logger.info(f"   python -c \"from speechbrain.pretrained import EncoderClassifier; EncoderClassifier.from_hparams(source='speechbrain/spkrec-ecapa-voxceleb', savedir='{self.model_cache_dir}')\"")
        self.logger.info("=" * 60)
    
    def _create_dummy_mean_var_file(self, filepath):
        """创建虚拟的mean_var_norm_emb.ckpt文件"""
        try:
            dummy_data = {
                'mean': torch.zeros(192),
                'std': torch.ones(192)
            }
            torch.save(dummy_data, filepath)
        except Exception as e:
            self.logger.warning(f"创建虚拟文件失败: {e}")
    
    def _validate_model(self) -> bool:
        """验证模型是否正确加载"""
        try:
            if self.classifier is None:
                self.logger.error("分类器为空")
                return False
            
            # 创建测试张量
            test_tensor = torch.randn(1, 16000).to(self.device)  # 1秒的测试音频
            
            # 尝试进行推理
            with torch.no_grad():
                embeddings = self.classifier.encode_batch(test_tensor)
                
            # 检查输出形状
            if embeddings.shape[-1] != 192:
                self.logger.error(f"模型输出维度错误: {embeddings.shape[-1]}, 期望: 192")
                return False
                
            self.logger.info("模型验证成功")
            return True
            
        except Exception as e:
            self.logger.error(f"模型验证失败: {e}")
            return False
    
    def load_audio(self, audio_path: str) -> Tuple[Optional[np.ndarray], Optional[int]]:
        """加载音频文件"""
        try:
            # 使用torchaudio加载音频
            waveform, sample_rate = torchaudio.load(audio_path)
            
            # 转换为numpy数组
            audio_data = waveform.numpy()
            
            # 如果是多声道，转换为单声道
            if len(audio_data.shape) > 1 and audio_data.shape[0] > 1:
                audio_data = np.mean(audio_data, axis=0)
            else:
                audio_data = audio_data.squeeze()
            
            return audio_data, sample_rate
            
        except Exception as e:
            self.logger.error(f"音频加载失败: {e}")
            return None, None
    
    def preprocess_audio(self, waveform: np.ndarray, sample_rate: int) -> Optional[torch.Tensor]:
        """音频预处理"""
        try:
            # 转换为单声道
            if len(waveform.shape) > 1:
                waveform = np.mean(waveform, axis=0)
            
            # 重采样到目标采样率
            if sample_rate != self.target_sr:
                if self.resampler is None:
                    self.resampler = torchaudio.transforms.Resample(
                        orig_freq=sample_rate, 
                        new_freq=self.target_sr
                    )
                
                waveform_tensor = torch.FloatTensor(waveform).unsqueeze(0)
                waveform = self.resampler(waveform_tensor).squeeze(0).numpy()
            
            # 确保音频长度适合模型输入（ECAPA通常需要2-3秒）
            target_length = self.target_sr * 3  # 3秒
            if len(waveform) > target_length:
                # 截取中间部分
                start = (len(waveform) - target_length) // 2
                waveform = waveform[start:start + target_length]
            elif len(waveform) < target_length:
                # 零填充
                padded = np.zeros(target_length)
                padded[:len(waveform)] = waveform
                waveform = padded
            
            # 转换为tensor，添加batch维度
            audio_tensor = torch.FloatTensor(waveform).unsqueeze(0).to(self.device)
            
            # 检查有效性
            if torch.isnan(audio_tensor).any() or torch.isinf(audio_tensor).any():
                self.logger.warning("音频包含无效值")
                return None
            
            return audio_tensor
            
        except Exception as e:
            self.logger.error(f"音频预处理失败: {e}")
            return None
    
    def extract_single(self, audio_path: str, vehicle_id: str = None) -> ExtractionResult:
        """
        提取单个音频文件的特征
        
        Args:
            audio_path: 音频文件路径
            vehicle_id: 车辆ID（可选）
            
        Returns:
            ExtractionResult: 提取结果对象
        """
        start_time = time.time()
        
        try:
            # 检查缓存
            if self.use_cache and self.cache:
                cached_features = self.cache.get(audio_path)
                if cached_features is not None:
                    self.logger.debug(f"使用缓存特征: {os.path.basename(audio_path)}")
                    return ExtractionResult(
                        file_path=audio_path,
                        success=True,
                        features=cached_features,
                        extraction_time=time.time() - start_time,
                        file_size=os.path.getsize(audio_path) if os.path.exists(audio_path) else 0
                    )
            
            # 获取音频时长（用于进度估算）
            try:
                duration = librosa.get_duration(path=audio_path)
            except:
                duration = 0.0
            
            # 提取特征
            features = self._extract_ecapa_features(audio_path)
            
            if features is not None:
                # 缓存特征
                if self.use_cache and self.cache:
                    self.cache.put(audio_path, features)
                self.logger.debug(f"特征提取完成并缓存: {os.path.basename(audio_path)}")
                
                return ExtractionResult(
                    file_path=audio_path,
                    success=True,
                    features=features,
                    extraction_time=time.time() - start_time,
                    file_size=os.path.getsize(audio_path) if os.path.exists(audio_path) else 0,
                    audio_duration=duration
                )
            else:
                self.logger.warning(f"特征提取失败: {os.path.basename(audio_path)}")
                return ExtractionResult(
                    file_path=audio_path,
                    success=False,
                    error_message="特征提取返回None",
                    extraction_time=time.time() - start_time,
                    file_size=os.path.getsize(audio_path) if os.path.exists(audio_path) else 0,
                    audio_duration=duration
                )
                
        except Exception as e:
            self.logger.error(f"特征提取异常 {os.path.basename(audio_path)}: {e}")
            return ExtractionResult(
                file_path=audio_path,
                success=False,
                error_message=str(e),
                extraction_time=time.time() - start_time,
                file_size=os.path.getsize(audio_path) if os.path.exists(audio_path) else 0
            )
    
    def _extract_ecapa_features(self, audio_path: str) -> Optional[Dict[str, np.ndarray]]:
        """
        从音频文件提取ECAPA-TDNN特征
        
        Args:
            audio_path: 音频文件路径
            
        Returns:
            Dict[str, np.ndarray]: 包含'fused_features'键的字典，或者None（如果失败）
        """
        if not self.model_loaded:
            self.logger.warning("ECAPA-TDNN模型未加载")
            return None
        
        try:
            filename = Path(audio_path).name
            
            # 加载音频
            waveform, sample_rate = self.load_audio(audio_path)
            if waveform is None:
                self.logger.error(f"音频加载失败: {filename}")
                return None
            
            # 预处理音频
            audio_tensor = self.preprocess_audio(waveform, sample_rate)
            if audio_tensor is None:
                self.logger.error(f"音频预处理失败: {filename}")
                return None
            
            # 提取特征
            with torch.no_grad():
                embeddings = self.classifier.encode_batch(audio_tensor)
            
            # 转换为numpy数组
            features = embeddings.squeeze().cpu().numpy()
            
            self.logger.debug(f"成功提取ECAPA特征: {filename}, 维度: {features.shape}")
            
            # 返回与相似度引擎兼容的格式
            return {
                'fused_features': features,
                'feature_type': 'ecapa_tdnn',
                'feature_dim': len(features)
            }
            
        except Exception as e:
            self.logger.error(f"特征提取失败 {filename}: {e}")
            return None
    
    def get_feature_dimension(self) -> int:
        """获取特征维度"""
        return 192  # ECAPA-TDNN输出192维特征
    
    def extract_features(self, audio_path: str, use_cache: bool = None) -> ExtractionResult:
        """
        提取音频特征（extract_single的别名，保持向后兼容）
        
        Args:
            audio_path: 音频文件路径
            use_cache: 是否使用缓存（覆盖全局设置）
        
        Returns:
            ExtractionResult: 提取结果
        """
        return self.extract_single(audio_path, use_cache)
    
    def extract_batch(
        self,
        audio_paths: List[str],
        progress_callback: Optional[callable] = None,
        use_cache: bool = None
    ) -> Tuple[List[ExtractionResult], BatchExtractionStats]:
        """
        批量提取音频特征（改为顺序处理避免并发问题）
        
        Args:
            audio_paths: 音频文件路径列表
            progress_callback: 进度回调函数 callback(completed, total, current_file)
            use_cache: 是否使用缓存
        
        Returns:
            Tuple[List[ExtractionResult], BatchExtractionStats]: 提取结果和统计信息
        """
        start_time = time.time()
        results = []
        stats = BatchExtractionStats()
        stats.total_files = len(audio_paths)
        
        self.logger.info(f"开始批量提取 {len(audio_paths)} 个音频文件的特征")
        
        # 改为顺序处理，避免并发问题
        completed = 0
        for i, path in enumerate(audio_paths):
            try:
                # 调用进度回调（处理前）
                if progress_callback:
                    try:
                        progress_callback(completed, stats.total_files, path)
                    except Exception as e:
                        self.logger.warning(f"进度回调失败: {e}")
                
                # 提取特征
                result = self.extract_single(path, use_cache)
                
                # 检查结果是否有效
                if result is None:
                    self.logger.error(f"extract_single返回None: {Path(path).name}")
                    result = ExtractionResult(
                        file_path=path,
                        success=False,
                        error_message="extract_single返回None"
                    )
                
                results.append(result)
                
                # 更新统计
                if result.success:
                    stats.successful_files += 1
                    stats.total_audio_duration += result.audio_duration
                    
                    # 检查是否命中缓存
                    if result.extraction_time < 0.1:  # 假设缓存命中时间很短
                        stats.cache_hits += 1
                    else:
                        stats.cache_misses += 1
                else:
                    stats.failed_files += 1
                
                completed += 1
                
                # 每处理5个文件进行一次垃圾回收
                if completed % 5 == 0:
                    gc.collect()
                
            except Exception as e:
                self.logger.error(f"处理文件时发生错误: {e}")
                results.append(ExtractionResult(
                    file_path=path,
                    success=False,
                    error_message=str(e)
                ))
                stats.failed_files += 1
                completed += 1
        
        # 计算统计信息
        stats.total_time = time.time() - start_time
        stats.avg_time_per_file = stats.total_time / stats.total_files if stats.total_files > 0 else 0
        
        self.logger.info(
            f"批量提取完成: {stats.successful_files}/{stats.total_files} 成功, "
            f"耗时: {stats.total_time:.2f}s, "
            f"平均: {stats.avg_time_per_file:.3f}s/文件"
        )
        
        return results, stats
    
    def extract_from_directory(
        self,
        directory: str,
        recursive: bool = True,
        audio_extensions: List[str] = None,
        progress_callback: Optional[callable] = None
    ) -> Tuple[List[ExtractionResult], BatchExtractionStats]:
        """
        从目录提取音频特征
        
        Args:
            directory: 音频文件目录
            recursive: 是否递归搜索子目录
            audio_extensions: 音频文件扩展名列表
            progress_callback: 进度回调函数
        
        Returns:
            Tuple[List[ExtractionResult], BatchExtractionStats]: 提取结果和统计信息
        """
        if audio_extensions is None:
            audio_extensions = ['.wav', '.mp3', '.flac', '.m4a', '.aac']
        
        # 使用公共工具函数获取音频文件
        audio_paths = file_manager.get_audio_files(directory, recursive, audio_extensions)
        
        self.logger.info(f"在目录 {directory} 中找到 {len(audio_paths)} 个音频文件")
        
        if not audio_paths:
            return [], BatchExtractionStats()
        
        return self.extract_batch(audio_paths, progress_callback)
    
    def get_cache_stats(self) -> Optional[Dict[str, Any]]:
        """获取缓存统计信息"""
        if self.cache:
            return self.cache.get_stats()
        return None
    
    def clear_cache(self):
        """清空缓存"""
        if self.cache:
            self.cache.clear()
    
    def set_batch_size(self, batch_size: int) -> None:
        """设置批量处理大小
        
        Args:
            batch_size: 批量处理大小，较小的值可避免系统卡住
        """
        if batch_size < 1:
            batch_size = 1
        elif batch_size > 32:
            batch_size = 32
            
        self.batch_size = batch_size
        self.logger.info(f"设置批量处理大小为: {batch_size}")
    
    def optimize_for_batch_processing(self):
        """优化批处理性能"""
        try:
            # 预热模型
            if hasattr(self, 'classifier') and self.classifier:
                dummy_input = torch.randn(1, 16000).to(self.device)
                with torch.no_grad():
                    _ = self.classifier.encode_batch(dummy_input)
            
            # 设置优化标志
            if str(self.device) == "cuda":
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.deterministic = False
            
            self.logger.info("批处理性能优化完成")
            
        except Exception as e:
            self.logger.warning(f"性能优化失败: {e}")

def create_optimized_extractor(
    use_cache: bool = True,
    max_workers: int = None,
    device: str = "auto",
    **kwargs
) -> OptimizedFeatureExtractor:
    """
    创建优化的特征提取器
    
    Args:
        use_cache: 是否使用缓存
        max_workers: 最大工作线程数
        device: 设备类型 ('cpu', 'cuda', 'auto')
        **kwargs: 其他参数
    
    Returns:
        OptimizedFeatureExtractor: 优化的特征提取器实例
    """
    return OptimizedFeatureExtractor(
        use_cache=use_cache,
        max_workers=max_workers,
        device=device,
        **kwargs
    )

def extract_features_from_paths(
    audio_paths: List[str],
    use_cache: bool = True,
    progress_callback: Optional[callable] = None
) -> Tuple[List[ExtractionResult], BatchExtractionStats]:
    """
    便捷函数：从路径列表提取特征
    
    Args:
        audio_paths: 音频文件路径列表
        use_cache: 是否使用缓存
        progress_callback: 进度回调函数
    
    Returns:
        Tuple[List[ExtractionResult], BatchExtractionStats]: 提取结果和统计信息
    """
    extractor = create_optimized_extractor(use_cache)
    return extractor.extract_batch(audio_paths, progress_callback)

def extract_features_from_directory(
    directory: str,
    use_cache: bool = True,
    recursive: bool = True,
    progress_callback: Optional[callable] = None
) -> Tuple[List[ExtractionResult], BatchExtractionStats]:
    """
    便捷函数：从目录提取特征
    
    Args:
        directory: 音频文件目录
        use_cache: 是否使用缓存
        recursive: 是否递归搜索
        progress_callback: 进度回调函数
    
    Returns:
        Tuple[List[ExtractionResult], BatchExtractionStats]: 提取结果和统计信息
    """
    extractor = create_optimized_extractor(use_cache)
    return extractor.extract_from_directory(directory, recursive, progress_callback=progress_callback)

if __name__ == "__main__":
    # 测试代码
    import argparse
    
    parser = argparse.ArgumentParser(description="优化特征提取器测试")
    parser.add_argument("--directory", "-d", help="测试目录")
    parser.add_argument("--file", "-f", help="测试单个文件")
    parser.add_argument("--no-cache", action="store_true", help="禁用缓存")
    
    args = parser.parse_args()
    
    def progress_callback(completed, total, current_file):
        print(f"进度: {completed}/{total} ({completed/total*100:.1f}%) - {Path(current_file).name}")
    
    extractor = create_optimized_extractor(use_cache=not args.no_cache)
    
    if args.file:
        print(f"测试单个文件: {args.file}")
        result = extractor.extract_single(args.file)
        print(f"结果: {'成功' if result.success else '失败'}")
        if result.success:
            print(f"特征形状: {result.features.shape}")
            print(f"提取时间: {result.extraction_time:.3f}s")
        else:
            print(f"错误: {result.error_message}")
    
    elif args.directory:
        print(f"测试目录: {args.directory}")
        results, stats = extractor.extract_from_directory(args.directory, progress_callback=progress_callback)
        print(f"\n统计结果:")
        print(f"总文件数: {stats.total_files}")
        print(f"成功: {stats.successful_files}")
        print(f"失败: {stats.failed_files}")
        print(f"总耗时: {stats.total_time:.2f}s")
        print(f"平均耗时: {stats.avg_time_per_file:.3f}s/文件")
        print(f"缓存命中: {stats.cache_hits}")
        print(f"缓存未命中: {stats.cache_misses}")
    
    else:
        print("请指定 --file 或 --directory 参数")