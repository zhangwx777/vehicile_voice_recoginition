# training/trainer.py

import torch
import gc
import time
from visualization import plot_training_history
from logger import logger
from early_stopping import EarlyStopping, GradientClipper
from settings import DEVICE_CONFIG, TRAINING_CONFIG


class ModelTrainer:
    """模型训练器"""

    def __init__(self, model, device=None):
        self.device = device if device else DEVICE_CONFIG.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        
        # 训练历史记录
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.learning_rates = []
        
        # 性能优化组件
        self.gradient_clipper = None
        self.early_stopping = None
        self.mixed_precision = DEVICE_CONFIG.get('mixed_precision', False) and torch.cuda.is_available()
        
        # 初始化混合精度训练
        if self.mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
            logger.info("混合精度训练已启用")
        else:
            self.scaler = None
        
        logger.info(f"模型训练器初始化完成，设备: {self.device}")

    def train(self, train_loader, val_loader, criterion, optimizer, scheduler=None, num_epochs=50,
              model_save_path='best_model.pth', enable_early_stopping=True, save_checkpoints=True):
        """训练模型"""
        # 初始化性能优化组件
        if enable_early_stopping:
            early_stopping_config = TRAINING_CONFIG.get('early_stopping', {})
            self.early_stopping = EarlyStopping(**early_stopping_config)
        
        gradient_clipping_config = TRAINING_CONFIG.get('gradient_clipping', {})
        if gradient_clipping_config.get('enabled', True):
            self.gradient_clipper = GradientClipper(
                max_norm=gradient_clipping_config.get('max_norm', 1.0)
            )
        
        best_val_acc = 0.0
        start_time = time.time()
        
        # 设置确定性计算
        if DEVICE_CONFIG.get('deterministic', True):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        
        logger.info(f"开始训练，总计 {num_epochs} 个轮次")
        
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            
            # 训练阶段
            train_loss, train_acc = self._train_epoch(train_loader, criterion, optimizer)
            
            # 验证阶段
            val_loss, val_acc = self.validate(val_loader, criterion)
            
            # 记录学习率
            current_lr = optimizer.param_groups[0]['lr']
            self.learning_rates.append(current_lr)
            
            # 更新学习率
            if scheduler:
                scheduler.step()
            
            # 保存最优模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self._save_model(model_save_path, epoch, val_acc, optimizer)
            
            # 保存检查点
            if save_checkpoints and (epoch + 1) % 10 == 0:
                checkpoint_path = model_save_path.replace('.pth', f'_epoch_{epoch+1}.pth')
                self._save_checkpoint(checkpoint_path, epoch, val_acc, optimizer, scheduler)
            
            # 计算轮次用时
            epoch_time = time.time() - epoch_start_time
            
            logger.info(
                f'Epoch [{epoch + 1}/{num_epochs}] | '
                f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | '
                f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}% | '
                f'LR: {current_lr:.2e} | Time: {epoch_time:.2f}s'
            )
            
            # 早停检查
            if self.early_stopping:
                if self.early_stopping(val_loss, self.model, epoch):
                    logger.info(f"早停触发，在第 {epoch + 1} 轮停止训练")
                    break
            
            # 内存清理
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        
        total_time = time.time() - start_time
        logger.info(f'训练完成。总用时: {total_time:.2f}s, 最优验证准确率: {best_val_acc:.2f}%')
        
        # 绘制训练历史
        self._plot_training_history()
        
        return best_val_acc
    
    def _train_epoch(self, train_loader, criterion, optimizer):
        """训练一个轮次"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            optimizer.zero_grad()
            
            # 混合精度训练
            if self.mixed_precision:
                with torch.cuda.amp.autocast():
                    outputs = self.model(inputs)
                    loss = criterion(outputs, labels)
                
                self.scaler.scale(loss).backward()
                
                # 梯度裁剪
                if self.gradient_clipper:
                    self.scaler.unscale_(optimizer)
                    grad_norm = self.gradient_clipper.clip_gradients(self.model)
                
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                
                # 梯度裁剪
                if self.gradient_clipper:
                    grad_norm = self.gradient_clipper.clip_gradients(self.model)
                
                optimizer.step()
            
            # 统计
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # 内存清理（每100个批次）
            if batch_idx % 100 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total
        
        self.train_losses.append(train_loss)
        self.train_accuracies.append(train_acc)
        
        return train_loss, train_acc

    def validate(self, val_loader, criterion):
        """验证模型"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                # 混合精度验证
                if self.mixed_precision:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(inputs)
                        loss = criterion(outputs, labels)
                else:
                    outputs = self.model(inputs)
                    loss = criterion(outputs, labels)

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        val_loss = running_loss / len(val_loader)
        val_acc = 100. * correct / total
        
        self.val_losses.append(val_loss)
        self.val_accuracies.append(val_acc)

        return val_loss, val_acc
    
    def _save_model(self, save_path, epoch, val_acc, optimizer):
        """保存模型"""
        try:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'epoch': epoch,
                'val_acc': val_acc,
                'optimizer_state_dict': optimizer.state_dict(),
            }, save_path)
            logger.info(f"模型已保存: {save_path} (验证准确率: {val_acc:.2f}%)")
        except Exception as e:
            logger.error(f"模型保存失败: {str(e)}")
    
    def _save_checkpoint(self, save_path, epoch, val_acc, optimizer, scheduler):
        """保存检查点"""
        try:
            checkpoint = {
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_acc': val_acc,
                'train_losses': self.train_losses,
                'val_losses': self.val_losses,
                'train_accuracies': self.train_accuracies,
                'val_accuracies': self.val_accuracies,
                'learning_rates': self.learning_rates,
            }
            
            if scheduler:
                checkpoint['scheduler_state_dict'] = scheduler.state_dict()
            
            if self.early_stopping:
                checkpoint['early_stopping_state'] = {
                    'best_score': self.early_stopping.best_score,
                    'counter': self.early_stopping.counter,
                    'best_epoch': self.early_stopping.best_epoch,
                }
            
            torch.save(checkpoint, save_path)
            logger.info(f"检查点已保存: {save_path}")
        except Exception as e:
            logger.error(f"检查点保存失败: {str(e)}")
    
    def _plot_training_history(self):
        """绘制训练历史"""
        try:
            plot_training_history(
                self.train_losses, self.val_losses,
                self.train_accuracies, self.val_accuracies,
                learning_rates=self.learning_rates
            )
        except Exception as e:
            logger.error(f"绘制训练历史失败: {str(e)}")