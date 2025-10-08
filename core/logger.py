#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一日志管理模块
提供系统级别的日志记录和管理功能
"""

# 直接导入避免循环依赖
import os
import sys
import time
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
import warnings
import numpy as np

import logging
import logging.handlers
from datetime import datetime

# 延迟导入配置，避免循环依赖
def get_logging_config():
    try:
        from config import LOGGING_CONFIG
        return LOGGING_CONFIG
    except ImportError:
        return DEFAULT_LOGGING_CONFIG

# 创建全局日志实例
logger = None

# 默认配置（作为备用）
DEFAULT_LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s | %(levelname)-8s | %(name)-25s | %(message)s',
    'file_enabled': True,
    'file_path': 'logs/system.log',
    'max_file_size': 10 * 1024 * 1024,  # 10MB
    'console_enabled': True
}

class ColoredFormatter(logging.Formatter):
    """彩色日志格式化器"""
    
    # ANSI颜色代码
    COLORS = {
        'DEBUG': '\033[36m',     # 青色
        'INFO': '\033[32m',      # 绿色
        'WARNING': '\033[33m',   # 黄色
        'ERROR': '\033[31m',     # 红色
        'CRITICAL': '\033[35m',  # 紫色
        'RESET': '\033[0m'       # 重置
    }
    
    def format(self, record):
        """格式化日志记录"""
        # 获取原始格式化结果
        log_message = super().format(record)
        
        # 添加颜色
        if record.levelname in self.COLORS:
            color = self.COLORS[record.levelname]
            reset = self.COLORS['RESET']
            log_message = f"{color}{log_message}{reset}"
        
        return log_message

class SystemLogger:
    """系统日志管理器"""
    
    def __init__(self, name: str = "VehicleRecognition"):
        """初始化日志管理器"""
        self.name = name
        self.logger = logging.getLogger(name)
        logging_config = get_logging_config()
        self.logger.setLevel(getattr(logging, logging_config['level']))
        
        # 避免重复添加处理器
        if not self.logger.handlers:
            self._setup_handlers()
    
    def _setup_handlers(self):
        """设置日志处理器"""
        logging_config = get_logging_config()
        
        # 控制台处理器
        if logging_config.get('console_enabled', True):
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.DEBUG)
            
            # 使用彩色格式化器
            console_formatter = ColoredFormatter(logging_config['format'])
            console_handler.setFormatter(console_formatter)
            
            self.logger.addHandler(console_handler)
        
        # 文件处理器
        if logging_config.get('file_enabled', True):
            log_file_path = Path(logging_config['file_path'])
            log_file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 使用RotatingFileHandler实现日志轮转
            file_handler = logging.handlers.RotatingFileHandler(
                log_file_path,
                maxBytes=logging_config.get('max_file_size', 10 * 1024 * 1024),
                backupCount=0,  # 不保留备份文件
                encoding='utf-8'
            )
            file_handler.setLevel(logging.DEBUG)
            
            # 文件使用普通格式化器（不需要颜色）
            file_formatter = logging.Formatter(logging_config['format'])
            file_handler.setFormatter(file_formatter)
            
            self.logger.addHandler(file_handler)
    
    def debug(self, message: str, **kwargs):
        """记录调试信息"""
        self.logger.debug(message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """记录信息"""
        self.logger.info(message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """记录警告"""
        self.logger.warning(message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """记录错误"""
        self.logger.error(message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """记录严重错误"""
        self.logger.critical(message, **kwargs)
    
    def exception(self, message: str, **kwargs):
        """记录异常信息（包含堆栈跟踪）"""
        self.logger.exception(message, **kwargs)
    
    def log_performance(self, operation: str, duration: float, **metrics):
        """记录性能指标"""
        message = f"性能统计 - {operation}: {duration:.3f}s"
        
        if metrics:
            metrics_str = ", ".join([f"{k}={v}" for k, v in metrics.items()])
            message += f" ({metrics_str})"
        
        self.info(message)
    
    def log_system_event(self, event: str, details: Dict[str, Any] = None):
        """记录系统事件"""
        message = f"系统事件: {event}"
        
        if details:
            details_str = ", ".join([f"{k}={v}" for k, v in details.items()])
            message += f" - {details_str}"
        
        self.info(message)


class ComponentLogger:
    """组件级别的日志管理器"""
    
    def __init__(self, component_name: str):
        """
        初始化组件日志管理器
        
        Args:
            component_name: 组件名称
        """
        self.component_name = component_name
        self._start_times: Dict[str, float] = {}
        # 使用全局logger实例
        self._logger = logger
    
    def debug(self, message: str, **kwargs):
        """记录调试信息"""
        self._logger.debug(f"[{self.component_name}] {message}", **kwargs)
    
    def info(self, message: str, **kwargs):
        """记录信息"""
        self._logger.info(f"[{self.component_name}] {message}", **kwargs)
    
    def warning(self, message: str, **kwargs):
        """记录警告"""
        self._logger.warning(f"[{self.component_name}] {message}", **kwargs)
    
    def error(self, message: str, **kwargs):
        """记录错误"""
        self._logger.error(f"[{self.component_name}] {message}", **kwargs)
    
    def critical(self, message: str, **kwargs):
        """记录严重错误"""
        self._logger.critical(f"[{self.component_name}] {message}", **kwargs)
    
    def exception(self, message: str, **kwargs):
        """记录异常信息"""
        self._logger.exception(f"[{self.component_name}] {message}", **kwargs)
    
    def start_timer(self, operation: str):
        """开始计时操作"""
        self._start_times[operation] = time.time()
        self.debug(f"开始操作: {operation}")
    
    def end_timer(self, operation: str, success: bool = True, **metrics):
        """结束计时操作并记录性能"""
        if operation in self._start_times:
            duration = time.time() - self._start_times[operation]
            status = "成功" if success else "失败"
            
            message = f"操作完成: {operation} - {status} ({duration:.3f}s)"
            if metrics:
                metrics_str = ", ".join([f"{k}={v}" for k, v in metrics.items()])
                message += f" [{metrics_str}]"
            
            self.info(message)
            del self._start_times[operation]
            return duration
        return None
    
    def log_operation(self, operation: str, success: bool = True, **details):
        """记录操作结果"""
        status = "成功" if success else "失败"
        message = f"操作: {operation} - {status}"
        
        if details:
            details_str = ", ".join([f"{k}={v}" for k, v in details.items()])
            message += f" [{details_str}]"
        
        self.info(message)
    
    def log_file_operation(self, operation: str, file_path: str, success: bool = True, **details):
        """记录文件操作"""
        filename = Path(file_path).name
        status = "成功" if success else "失败"
        message = f"文件操作: {operation} - {filename} - {status}"
        
        if details:
            details_str = ", ".join([f"{k}={v}" for k, v in details.items()])
            message += f" [{details_str}]"
        
        self.info(message)


# 初始化全局日志实例
logger = SystemLogger()

# 预定义的组件日志实例
recognition_logger = ComponentLogger("识别引擎")
feature_logger = ComponentLogger("特征提取")
database_logger = ComponentLogger("数据库")
preprocessing_logger = ComponentLogger("预处理")
exception_logger = ComponentLogger("异常处理")

def setup_component_loggers():
    """设置所有组件日志实例"""
    return {
        'recognition': recognition_logger,
        'feature': feature_logger,
        'database': database_logger,
        'preprocessing': preprocessing_logger
    }


def log_system_startup():
    """记录系统启动信息"""
    logger.info("=" * 80)
    logger.info("车辆声纹识别系统启动")
    logger.info("=" * 80)


def log_system_shutdown():
    """记录系统关闭信息"""
    logger.info("=" * 80)
    logger.info("车辆声纹识别系统关闭")
    logger.info("=" * 80)


def log_module_loaded(module_name: str, version: str = ""):
    """记录模块加载信息"""
    if version:
        logger.info(f"模块加载: {module_name} v{version}")
    else:
        logger.info(f"模块加载: {module_name}")


def log_config_loaded(config_name: str, config_path: str):
    """记录配置加载信息"""
    logger.info(f"配置加载: {config_name} -> {config_path}")


# 兼容性函数
def setup_logger(name='vehicle_voice_recognition', log_dir='logs', level=logging.INFO):
    """设置日志记录器（兼容性函数）"""
    return SystemLogger(name).logger

# 导出主要接口
__all__ = [
    'SystemLogger', 
    'ComponentLogger',
    'setup_logger', 
    'logger',
    'recognition_logger',
    'feature_logger',
    'database_logger',
    'preprocessing_logger',
    'exception_logger',
    'setup_component_loggers',
    'log_system_startup',
    'log_system_shutdown',
    'log_module_loaded',
    'log_config_loaded'
]