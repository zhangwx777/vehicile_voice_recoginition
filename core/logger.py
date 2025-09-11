#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一日志管理模块
提供系统级别的日志记录和管理功能
"""

import os
import sys
import logging
import logging.handlers
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

# 导入配置
try:
    from config import LOGGING_CONFIG
except ImportError:
    # 默认配置
    LOGGING_CONFIG = {
        'level': 'INFO',
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        'file_enabled': True,
        'file_path': 'logs/system.log',
        'max_file_size': 10 * 1024 * 1024,  # 10MB
        'backup_count': 5,
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
        self.logger.setLevel(getattr(logging, LOGGING_CONFIG['level']))
        
        # 避免重复添加处理器
        if not self.logger.handlers:
            self._setup_handlers()
    
    def _setup_handlers(self):
        """设置日志处理器"""
        # 控制台处理器
        if LOGGING_CONFIG.get('console_enabled', True):
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.DEBUG)
            
            # 使用彩色格式化器
            console_formatter = ColoredFormatter(LOGGING_CONFIG['format'])
            console_handler.setFormatter(console_formatter)
            
            self.logger.addHandler(console_handler)
        
        # 文件处理器
        if LOGGING_CONFIG.get('file_enabled', True):
            log_file_path = Path(LOGGING_CONFIG['file_path'])
            log_file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 使用RotatingFileHandler实现日志轮转
            file_handler = logging.handlers.RotatingFileHandler(
                log_file_path,
                maxBytes=LOGGING_CONFIG.get('max_file_size', 10 * 1024 * 1024),
                backupCount=LOGGING_CONFIG.get('backup_count', 5),
                encoding='utf-8'
            )
            file_handler.setLevel(logging.DEBUG)
            
            # 文件使用普通格式化器（不需要颜色）
            file_formatter = logging.Formatter(LOGGING_CONFIG['format'])
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

# 兼容性函数
def setup_logger(name='vehicle_voice_recognition', log_dir='logs', level=logging.INFO):
    """设置日志记录器（兼容性函数）"""
    return SystemLogger(name).logger

# 创建全局日志实例
logger = SystemLogger()

# 导出主要接口
__all__ = ['SystemLogger', 'setup_logger', 'logger']