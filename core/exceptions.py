"""
统一异常处理机制
提供系统级别的异常定义和处理工具
"""

# 使用公共工具模块统一导入
from .common_utils import (
    os, sys, time, json, Path,
    List, Dict, Optional, Tuple, Any, Union,
    dataclass, asdict, warnings, np,
    config_manager, file_manager, logger_factory, TimerContext
)

# 添加缺少的Callable导入
from typing import Callable

import traceback
from functools import wraps

exception_logger = logger_factory.get_logger("异常处理")


class VehicleVoiceRecognitionError(Exception):
    """车辆语音识别系统基础异常类"""
    
    def __init__(self, message: str, error_code: str = None, details: dict = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or "UNKNOWN_ERROR"
        self.details = details or {}
        
    def __str__(self):
        return f"[{self.error_code}] {self.message}"


# 向后兼容别名
VehicleRecognitionError = VehicleVoiceRecognitionError


class AudioProcessingError(VehicleVoiceRecognitionError):
    """音频处理相关异常"""
    
    def __init__(self, message: str, audio_path: str = None, **kwargs):
        super().__init__(message, error_code="AUDIO_PROCESSING_ERROR", **kwargs)
        self.audio_path = audio_path


class FeatureExtractionError(VehicleVoiceRecognitionError):
    """特征提取相关异常"""
    
    def __init__(self, message: str, feature_type: str = None, **kwargs):
        super().__init__(message, error_code="FEATURE_EXTRACTION_ERROR", **kwargs)
        self.feature_type = feature_type


class ModelLoadingError(VehicleVoiceRecognitionError):
    """模型加载相关异常"""
    
    def __init__(self, message: str, model_path: str = None, **kwargs):
        super().__init__(message, error_code="MODEL_LOADING_ERROR", **kwargs)
        self.model_path = model_path


class SimilaritySearchError(VehicleVoiceRecognitionError):
    """相似度搜索相关异常"""
    
    def __init__(self, message: str, query_path: str = None, **kwargs):
        super().__init__(message, error_code="SIMILARITY_SEARCH_ERROR", **kwargs)
        self.query_path = query_path





class DataValidationError(VehicleVoiceRecognitionError):
    """数据验证相关异常"""
    
    def __init__(self, message: str, data_type: str = None, **kwargs):
        super().__init__(message, error_code="DATA_VALIDATION_ERROR", **kwargs)
        self.data_type = data_type


def handle_exceptions(
    default_return=None, 
    log_error=True, 
    reraise=False,
    exception_types=(Exception,),
    context: str = None,
    include_traceback: bool = True
):
    """
    增强的异常处理装饰器
    
    Args:
        default_return: 异常时的默认返回值
        log_error: 是否记录错误日志
        reraise: 是否重新抛出异常
        exception_types: 要捕获的异常类型
        context: 异常上下文信息
        include_traceback: 是否包含详细的堆栈跟踪
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except exception_types as e:
                if log_error:
                    # 标准化异常日志格式
                    context_info = context or f"函数 {func.__module__}.{func.__name__}"
                    exception_logger.error(
                        f"[异常处理] {context_info} 执行失败: {type(e).__name__}: {str(e)}"
                    )
                    
                    if include_traceback:
                        exception_logger.debug(f"详细错误信息:\n{traceback.format_exc()}")
                
                if reraise:
                    raise
                    
                return default_return
        return wrapper
    return decorator


def async_handle_exceptions(
    default_return=None, 
    log_error=True, 
    reraise=False,
    exception_types=(Exception,),
    context: str = None,
    include_traceback: bool = True
):
    """
    异步函数的异常处理装饰器
    
    Args:
        default_return: 异常时的默认返回值
        log_error: 是否记录错误日志
        reraise: 是否重新抛出异常
        exception_types: 要捕获的异常类型
        context: 异常上下文信息
        include_traceback: 是否包含详细的堆栈跟踪
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except exception_types as e:
                if log_error:
                    # 标准化异常日志格式
                    context_info = context or f"异步函数 {func.__module__}.{func.__name__}"
                    exception_logger.error(
                        f"[异常处理] {context_info} 执行失败: {type(e).__name__}: {str(e)}"
                    )
                    
                    if include_traceback:
                        exception_logger.debug(f"详细错误信息:\n{traceback.format_exc()}")
                
                if reraise:
                    raise
                    
                return default_return
        return wrapper
    return decorator


def retry_on_exception(
    max_retries: int = 3,
    delay: float = 1.0,
    backoff_factor: float = 2.0,
    exception_types=(Exception,),
    log_retries: bool = True
):
    """
    重试装饰器，在异常时自动重试
    
    Args:
        max_retries: 最大重试次数
        delay: 初始延迟时间（秒）
        backoff_factor: 延迟时间的倍增因子
        exception_types: 触发重试的异常类型
        log_retries: 是否记录重试日志
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            current_delay = delay
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exception_types as e:
                    last_exception = e
                    
                    if attempt < max_retries:
                        if log_retries:
                            exception_logger.warning(
                                f"[重试机制] 函数 {func.__name__} 第 {attempt + 1} 次执行失败，"
                                f"{current_delay:.1f}秒后重试: {str(e)}"
                            )
                        
                        time.sleep(current_delay)
                        current_delay *= backoff_factor
                    else:
                        if log_retries:
                            exception_logger.error(
                                f"[重试机制] 函数 {func.__name__} 重试 {max_retries} 次后仍然失败: {str(e)}"
                            )
                        raise last_exception
            
            return None
        return wrapper
    return decorator


def safe_execute(
    func: Callable, 
    *args, 
    default_return=None, 
    log_error=True,
    error_message: str = None,
    context: str = None,
    include_traceback: bool = True,
    **kwargs
) -> Any:
    """
    安全执行函数，捕获异常并返回默认值
    
    Args:
        func: 要执行的函数
        *args: 函数参数
        default_return: 异常时的默认返回值
        log_error: 是否记录错误日志
        error_message: 自定义错误消息
        context: 异常上下文信息
        include_traceback: 是否包含详细的堆栈跟踪
        **kwargs: 函数关键字参数
    
    Returns:
        函数执行结果或默认值
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        if log_error:
            context_info = context or f"安全执行函数 {func.__name__}"
            message = error_message or f"{context_info} 时发生错误"
            exception_logger.error(f"[安全执行] {message}: {type(e).__name__}: {str(e)}")
            
            if include_traceback:
                exception_logger.debug(f"详细错误信息:\n{traceback.format_exc()}")
        
        return default_return


def validate_audio_path(audio_path: str) -> str:
    """验证音频文件路径"""
    import os
    
    if not audio_path:
        raise AudioProcessingError("音频路径不能为空", audio_path=audio_path)
    
    if not os.path.exists(audio_path):
        raise AudioProcessingError(f"音频文件不存在: {audio_path}", audio_path=audio_path)
    
    if not os.path.isfile(audio_path):
        raise AudioProcessingError(f"路径不是文件: {audio_path}", audio_path=audio_path)
    
    # 检查文件扩展名
    valid_extensions = ['.wav', '.mp3', '.flac', '.m4a', '.ogg']
    file_ext = os.path.splitext(audio_path)[1].lower()
    if file_ext not in valid_extensions:
        raise AudioProcessingError(
            f"不支持的音频格式: {file_ext}, 支持的格式: {valid_extensions}",
            audio_path=audio_path
        )
    
    return audio_path


def validate_features(features, feature_type: str = "unknown"):
    """验证特征数据"""
    import numpy as np
    
    if features is None:
        raise FeatureExtractionError(f"特征数据为空", feature_type=feature_type)
    
    if not isinstance(features, np.ndarray):
        raise FeatureExtractionError(
            f"特征数据类型错误: {type(features)}, 期望: numpy.ndarray",
            feature_type=feature_type
        )
    
    if features.size == 0:
        raise FeatureExtractionError(f"特征数据长度为0", feature_type=feature_type)
    
    if np.any(np.isnan(features)):
        raise FeatureExtractionError(f"特征数据包含NaN值", feature_type=feature_type)
    
    if np.any(np.isinf(features)):
        raise FeatureExtractionError(f"特征数据包含无穷大值", feature_type=feature_type)
    
    return features


def log_system_error(error: Exception, context: str = "", include_traceback: bool = True):
    """
    记录系统错误日志
    
    Args:
        error: 异常对象
        context: 错误上下文
        include_traceback: 是否包含详细的堆栈跟踪
    """
    error_info = {
        'type': type(error).__name__,
        'message': str(error),
        'context': context,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    exception_logger.error(f"[系统错误] {context}: {error_info['type']}: {error_info['message']}")
    
    if include_traceback:
        exception_logger.debug(f"详细错误信息:\n{traceback.format_exc()}")
    
    return error_info


class ExceptionContext:
    """异常上下文管理器"""
    
    def __init__(self, context_name: str, log_errors: bool = True):
        self.context_name = context_name
        self.log_errors = log_errors
        self.exception_occurred = False
        self.exception_info = None
    
    def __enter__(self):
        exception_logger.debug(f"进入异常上下文: {self.context_name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.exception_occurred = True
            self.exception_info = {
                "type": exc_type.__name__,
                "value": str(exc_val),
                "context": self.context_name
            }
            
            if self.log_errors:
                log_system_error(exc_val, self.context_name)
        
        exception_logger.debug(f"退出异常上下文: {self.context_name}")
        return False  # 不抑制异常