# training/trainer.py

import torch
import gc
import time
from utils.visualization import plot_training_history
from core.logger import logger
from training.early_stopping import EarlyStopping, GradientClipper
from core.settings import DEVICE_CONFIG, TRAINING_CONFIG


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
                max_norm=gradient_clipping_config.get('max_norm', 1.0),
                norm_type=gradient_clipping_config.get('norm_type', 2.0)
            )
        
        best_val_loss = float('inf')
        start_time = time.time()
        
        logger.info(f"开始训练，共 {num_epochs} 个epoch")
        
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            
            # 训练阶段
            train_loss, train_acc = self._train_epoch(train_loader, criterion, optimizer)
            
            # 验证阶段
            val_loss, val_acc = self._validate_epoch(val_loader, criterion)
            
            # 记录学习率
            current_lr = optimizer.param_groups[0]['lr']
            self.learning_rates.append(current_lr)
            
            # 更新学习率调度器
            if scheduler:
                if hasattr(scheduler, 'step'):
                    if 'ReduceLROnPlateau' in str(type(scheduler)):
                        scheduler.step(val_loss)
                    else:
                        scheduler.step()
            
            # 记录训练历史
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            epoch_time = time.time() - epoch_start_time
            
            logger.info(
                f"Epoch {epoch+1}/{num_epochs} - "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, "
                f"LR: {current_lr:.6f}, Time: {epoch_time:.2f}s"
            )
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_acc': val_acc
                }, model_save_path)
                logger.info(f"保存最佳模型到 {model_save_path}")
            
            # 早停检查
            if self.early_stopping and self.early_stopping(val_loss):
                logger.info(f"早停触发，在第 {epoch+1} 个epoch停止训练")
                break
            
            # 内存清理
            if epoch % 10 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        total_time = time.time() - start_time
        logger.info(f"训练完成，总用时: {total_time:.2f}秒")
        
        # 最终内存清理
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("GPU缓存已清理")
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
            'learning_rates': self.learning_rates,
            'best_val_loss': best_val_loss
        }

    def _train_epoch(self, train_loader, criterion, optimizer):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            optimizer.zero_grad()
            
            if self.mixed_precision:
                with torch.cuda.amp.autocast():
                    output = self.model(data)
                    loss = criterion(output, target)
                
                self.scaler.scale(loss).backward()
                
                if self.gradient_clipper:
                    self.scaler.unscale_(optimizer)
                    self.gradient_clipper.clip_gradients(self.model)
                
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                
                if self.gradient_clipper:
                    self.gradient_clipper.clip_gradients(self.model)
                
                optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            # 清理中间变量以释放内存
            del data, target, output, loss, pred
            
            # 每50个batch清理一次GPU缓存
            if batch_idx % 50 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy

    def _validate_epoch(self, val_loader, criterion):
        """验证一个epoch"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                if self.mixed_precision:
                    with torch.cuda.amp.autocast():
                        output = self.model(data)
                        loss = criterion(output, target)
                else:
                    output = self.model(data)
                    loss = criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
                
                # 清理中间变量以释放内存
                del data, target, output, loss, pred
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy

    def save_training_plot(self, save_path='training_history.png'):
        """保存训练历史图表"""
        if len(self.train_losses) > 0:
            history_data = {
                'train_loss': self.train_losses,
                'val_loss': self.val_losses,
                'train_acc': self.train_accuracies,
                'val_acc': self.val_accuracies,
                'learning_rates': self.learning_rates
            }
            plot_training_history(
                history_data,
                save_path=save_path,
                title='个体识别训练历史'
            )
            logger.info(f"训练历史图表已保存到 {save_path}")
        else:
            logger.warning("没有训练历史数据可保存")

    def get_training_summary(self):
        """获取训练摘要"""
        if len(self.train_losses) == 0:
            return "没有训练数据"
        
        best_train_acc = max(self.train_accuracies)
        best_val_acc = max(self.val_accuracies)
        final_train_loss = self.train_losses[-1]
        final_val_loss = self.val_losses[-1]
        
        summary = f"""
训练摘要:
- 训练轮数: {len(self.train_losses)}
- 最佳训练准确率: {best_train_acc:.4f}
- 最佳验证准确率: {best_val_acc:.4f}
- 最终训练损失: {final_train_loss:.4f}
- 最终验证损失: {final_val_loss:.4f}
        """
        
        return summary