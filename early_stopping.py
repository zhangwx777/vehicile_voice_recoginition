# training/early_stopping.py

import numpy as np
import torch
from logger import logger


class EarlyStopping:
    """早停机制实现类"""
    
    def __init__(self, patience=10, min_delta=0.001, restore_best_weights=True, 
                 monitor='val_loss', mode='min', verbose=True):
        """
        初始化早停机制
        
        Args:
            patience: 容忍的验证指标不改善的轮次
            min_delta: 最小改善幅度
            restore_best_weights: 是否恢复最佳权重
            monitor: 监控的指标名称
            mode: 'min' 表示指标越小越好，'max' 表示指标越大越好
            verbose: 是否输出详细信息
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.monitor = monitor
        self.mode = mode
        self.verbose = verbose
        
        # 初始化状态
        self.best_score = None
        self.best_weights = None
        self.counter = 0
        self.early_stop = False
        self.best_epoch = 0
        
        # 根据模式设置比较函数
        if mode == 'min':
            self.is_better = lambda current, best: current < best - min_delta
            self.best_score = float('inf')
        elif mode == 'max':
            self.is_better = lambda current, best: current > best + min_delta
            self.best_score = float('-inf')
        else:
            raise ValueError(f"模式 '{mode}' 不支持，请使用 'min' 或 'max'")
        
        logger.info(f"早停机制初始化: patience={patience}, min_delta={min_delta}, monitor={monitor}, mode={mode}")
    
    def __call__(self, current_score, model, epoch):
        """
        检查是否应该早停
        
        Args:
            current_score: 当前轮次的监控指标值
            model: 当前模型
            epoch: 当前轮次
            
        Returns:
            bool: 是否应该早停
        """
        if self.is_better(current_score, self.best_score):
            # 指标有改善
            self.best_score = current_score
            self.best_epoch = epoch
            self.counter = 0
            
            # 保存最佳权重
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy() if hasattr(model, 'state_dict') else None
                
            if self.verbose:
                logger.info(f"验证{self.monitor}改善: {current_score:.6f}, 重置计数器")
        else:
            # 指标没有改善
            self.counter += 1
            
            if self.verbose:
                logger.info(f"验证{self.monitor}未改善: {current_score:.6f}, 计数器: {self.counter}/{self.patience}")
            
            # 检查是否达到早停条件
            if self.counter >= self.patience:
                self.early_stop = True
                
                if self.verbose:
                    logger.info(f"早停触发! 最佳{self.monitor}: {self.best_score:.6f} (第{self.best_epoch}轮)")
                
                # 恢复最佳权重
                if self.restore_best_weights and self.best_weights is not None:
                    model.load_state_dict(self.best_weights)
                    if self.verbose:
                        logger.info("已恢复最佳模型权重")
        
        return self.early_stop
    
    def get_best_score(self):
        """获取最佳分数"""
        return self.best_score
    
    def get_best_epoch(self):
        """获取最佳轮次"""
        return self.best_epoch
    
    def reset(self):
        """重置早停机制状态"""
        self.best_score = float('inf') if self.mode == 'min' else float('-inf')
        self.best_weights = None
        self.counter = 0
        self.early_stop = False
        self.best_epoch = 0
        
        if self.verbose:
            logger.info("早停机制状态已重置")


class GradientClipper:
    """梯度裁剪工具"""
    
    def __init__(self, max_norm=1.0, norm_type=2.0, enabled=True):
        """
        初始化梯度裁剪器
        
        Args:
            max_norm: 梯度的最大范数
            norm_type: 范数类型
            enabled: 是否启用梯度裁剪
        """
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.enabled = enabled
        
        if enabled:
            logger.info(f"梯度裁剪已启用: max_norm={max_norm}, norm_type={norm_type}")
    
    def clip_gradients(self, model):
        """
        裁剪模型梯度
        
        Args:
            model: PyTorch模型
            
        Returns:
            float: 梯度范数
        """
        if not self.enabled:
            return 0.0
            
        try:
            # 计算梯度范数并进行裁剪
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), 
                max_norm=self.max_norm, 
                norm_type=self.norm_type
            )
            return grad_norm.item()
        except Exception as e:
            logger.warning(f"梯度裁剪失败: {str(e)}")
            return 0.0
