# training/early_stopping.py

import numpy as np
import torch
from core.logger import logger


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
    
    def __call__(self, current_score, model=None, epoch=None):
        """
        检查是否应该早停
        
        Args:
            current_score: 当前轮次的监控指标值
            model: 模型对象（用于保存最佳权重）
            epoch: 当前轮次
            
        Returns:
            bool: 是否应该早停
        """
        if self.is_better(current_score, self.best_score):
            # 发现更好的结果
            self.best_score = current_score
            self.best_epoch = epoch if epoch is not None else 0
            self.counter = 0
            
            # 保存最佳权重
            if self.restore_best_weights and model is not None:
                self.best_weights = {k: v.clone() for k, v in model.state_dict().items()}
            
            if self.verbose:
                logger.info(f"发现更好的 {self.monitor}: {current_score:.6f} (epoch {self.best_epoch})")
                
        else:
            # 没有改善
            self.counter += 1
            if self.verbose:
                logger.info(f"{self.monitor} 没有改善: {current_score:.6f}, 计数器: {self.counter}/{self.patience}")
            
            # 检查是否达到早停条件
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    logger.info(f"早停触发！最佳 {self.monitor}: {self.best_score:.6f} (epoch {self.best_epoch})")
                
                # 恢复最佳权重
                if self.restore_best_weights and model is not None and self.best_weights is not None:
                    model.load_state_dict(self.best_weights)
                    if self.verbose:
                        logger.info("已恢复最佳权重")
        
        return self.early_stop
    
    def reset(self):
        """重置早停状态"""
        self.best_score = float('inf') if self.mode == 'min' else float('-inf')
        self.best_weights = None
        self.counter = 0
        self.early_stop = False
        self.best_epoch = 0
        
        if self.verbose:
            logger.info("早停机制已重置")


class GradientClipper:
    """梯度裁剪类"""
    
    def __init__(self, max_norm=1.0, norm_type=2.0, verbose=False):
        """
        初始化梯度裁剪器
        
        Args:
            max_norm: 梯度的最大范数
            norm_type: 范数类型
            verbose: 是否输出详细信息
        """
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.verbose = verbose
        
        logger.info(f"梯度裁剪器初始化: max_norm={max_norm}, norm_type={norm_type}")
    
    def clip_gradients(self, model):
        """
        裁剪模型梯度
        
        Args:
            model: PyTorch模型
            
        Returns:
            float: 裁剪前的梯度范数
        """
        # 计算梯度范数
        total_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), 
            self.max_norm, 
            norm_type=self.norm_type
        )
        
        if self.verbose and total_norm > self.max_norm:
            logger.debug(f"梯度被裁剪: {total_norm:.6f} -> {self.max_norm}")
        
        return total_norm
    
    def __call__(self, model):
        """使对象可调用"""
        return self.clip_gradients(model)
