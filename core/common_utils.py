#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
公共工具模块
整合项目中重复使用的工具函数和常用导入
"""

# 标准库导入 - 统一管理常用导入
import os
import sys
import time
import json
import hashlib
import pickle
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import warnings

# 第三方库导入
import numpy as np

# 项目配置导入
from config import (
    PATH_CONFIG, 
    AUDIO_CONFIG, 
    SIMILARITY_CONFIG, 
    LOGGING_CONFIG,
    ECAPA_FEATURE_CONFIG
)
from .logger import ComponentLogger


class ConfigManager:
    """统一配置管理器"""
    
    _instance = None
    _configs = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._configs = {
                'path': PATH_CONFIG,
                'audio': AUDIO_CONFIG,
                'similarity': SIMILARITY_CONFIG,
                'logging': LOGGING_CONFIG,
                'ecapa': ECAPA_FEATURE_CONFIG
            }
        return cls._instance
    
    def get(self, config_type: str, key: str = None, default=None):
        """获取配置值"""
        if config_type not in self._configs:
            return default
        
        config = self._configs[config_type]
        if key is None:
            return config
        
        return config.get(key, default)
    
    def get_path(self, key: str, default: str = None) -> str:
        """获取路径配置"""
        return self.get('path', key, default)
    
    def get_audio_config(self, key: str = None, default=None):
        """获取音频配置"""
        return self.get('audio', key, default)
    
    def get_similarity_config(self, key: str = None, default=None):
        """获取相似度配置"""
        return self.get('similarity', key, default)


class FileManager:
    """统一文件管理器"""
    
    @staticmethod
    def ensure_dir(path: Union[str, Path]) -> Path:
        """确保目录存在"""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    @staticmethod
    def safe_load_json(file_path: Union[str, Path], default=None) -> Any:
        """安全加载JSON文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            warnings.warn(f"加载JSON文件失败 {file_path}: {e}")
            return default if default is not None else {}
    
    @staticmethod
    def safe_save_json(file_path: Union[str, Path], data: Any, indent: int = 2) -> bool:
        """安全保存JSON文件"""
        try:
            file_path = Path(file_path)
            FileManager.ensure_dir(file_path.parent)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=indent, ensure_ascii=False)
            return True
        except Exception as e:
            warnings.warn(f"保存JSON文件失败 {file_path}: {e}")
            return False
    
    @staticmethod
    def safe_load_pickle(file_path: Union[str, Path], default=None) -> Any:
        """安全加载pickle文件"""
        try:
            with open(file_path, 'rb') as f:
                return pickle.load(f)
        except (FileNotFoundError, pickle.PickleError) as e:
            warnings.warn(f"加载pickle文件失败 {file_path}: {e}")
            return default
    
    @staticmethod
    def safe_save_pickle(data: Any, file_path: Union[str, Path]) -> bool:
        """安全保存pickle文件"""
        try:
            file_path = Path(file_path)
            FileManager.ensure_dir(file_path.parent)
            
            with open(file_path, 'wb') as f:
                pickle.dump(data, f)
            return True
        except Exception as e:
            warnings.warn(f"保存pickle文件失败 {file_path}: {e}")
            return False
    
    @staticmethod
    def get_file_hash(file_path: Union[str, Path]) -> str:
        """计算文件哈希值"""
        hash_md5 = hashlib.md5()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            
            hash_result = hash_md5.hexdigest()
            return hash_result
        except Exception as e:
            try:
                logger = logger_factory.get_feature_logger()
                logger.error(f"哈希计算失败: {e}")
            except Exception as e:
                warnings.warn(f"哈希计算失败: {e}")
            return ""
    
    @staticmethod
    def file_exists(file_path: Union[str, Path]) -> bool:
        """检查文件是否存在"""
        try:
            return Path(file_path).exists()
        except Exception:
            return False


class LoggerFactory:
    """日志工厂类 - 统一管理组件日志"""
    
    _loggers = {}
    
    @classmethod
    def get_logger(cls, component_name: str) -> ComponentLogger:
        """获取或创建组件日志器"""
        if component_name not in cls._loggers:
            cls._loggers[component_name] = ComponentLogger(component_name)
        return cls._loggers[component_name]
    
    @classmethod
    def get_feature_logger(cls) -> ComponentLogger:
        """获取特征提取日志器"""
        return cls.get_logger("特征提取")
    
    @classmethod
    def get_database_logger(cls) -> ComponentLogger:
        """获取数据库日志器"""
        return cls.get_logger("数据库管理")
    
    @classmethod
    def get_similarity_logger(cls) -> ComponentLogger:
        """获取相似度引擎日志器"""
        return cls.get_logger("相似度引擎")
    
    @classmethod
    def get_recognition_logger(cls) -> ComponentLogger:
        """获取识别引擎日志器"""
        return cls.get_logger("识别引擎")


class TimerContext:
    """计时上下文管理器"""
    
    def __init__(self, logger: ComponentLogger, operation: str):
        self.logger = logger
        self.operation = operation
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        self.logger.info(f"开始 {self.operation}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        if exc_type is None:
            self.logger.info(f"完成 {self.operation}，耗时: {duration:.3f}s")
        else:
            self.logger.error(f"失败 {self.operation}，耗时: {duration:.3f}s，错误: {exc_val}")


def get_audio_files(directory: Union[str, Path], 
                   extensions: List[str] = None,
                   recursive: bool = True) -> List[Path]:
    """获取目录中的音频文件"""
    if extensions is None:
        extensions = ['.wav', '.mp3', '.flac', '.m4a', '.aac']
    
    directory = Path(directory)
    if not directory.exists():
        return []
    
    audio_files = []
    pattern = "**/*" if recursive else "*"
    
    for ext in extensions:
        audio_files.extend(directory.glob(f"{pattern}{ext}"))
        audio_files.extend(directory.glob(f"{pattern}{ext.upper()}"))
    
    return sorted(audio_files)


def format_duration(seconds: float) -> str:
    """格式化时间长度"""
    if seconds < 60:
        return f"{seconds:.1f}秒"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}分钟"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}小时"


def format_file_size(size_bytes: int) -> str:
    """格式化文件大小"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f}{unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f}TB"


# 全局实例
config_manager = ConfigManager()
file_manager = FileManager()
logger_factory = LoggerFactory()

# 导出的公共接口
__all__ = [
    # 类
    'ConfigManager',
    'FileManager', 
    'LoggerFactory',
    'TimerContext',
    
    # 实例
    'config_manager',
    'file_manager',
    'logger_factory',
    
    # 工具函数
    'get_audio_files',
    'format_duration',
    'format_file_size',
    
    # 常用导入
    'os', 'sys', 'time', 'json', 'hashlib', 'pickle', 'shutil',
    'Path', 'Dict', 'List', 'Optional', 'Any', 'Tuple', 'Union',
    'dataclass', 'asdict', 'datetime', 'warnings', 'np'
]