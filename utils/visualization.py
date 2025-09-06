# utils/visualization.py

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import librosa
import librosa.display
from sklearn.metrics import confusion_matrix, classification_report
import os
import logging

# 设置日志
logger = logging.getLogger(__name__)

# 设置matplotlib使用英文字体，避免中文字体缺失警告
plt.rcParams["font.family"] = ["DejaVu Sans", "Arial", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False  # Fix minus sign display


def plot_training_history(history, save_path='results/training_history.png', 
                          figsize=(12, 6), title='Training and Validation Metrics'):
    """绘制训练历史，支持学习率曲线
    
    Args:
        history: 可以是字典格式 {'train_loss': [], 'val_loss': [], ...} 
                或者是旧格式的参数列表
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 处理不同的输入格式
    if isinstance(history, dict):
        train_losses = history.get('train_loss', [])
        val_losses = history.get('val_loss', [])
        train_accuracies = history.get('train_acc', [])
        val_accuracies = history.get('val_acc', [])
        learning_rates = history.get('learning_rates', [])
    else:
        # 兼容旧格式（如果第一个参数不是字典）
        train_losses = history
        val_losses = save_path if isinstance(save_path, list) else []
        train_accuracies = figsize if isinstance(figsize, list) else []
        val_accuracies = title if isinstance(title, list) else []
        learning_rates = []
        save_path = 'results/training_history.png'
        figsize = (12, 6)
        title = 'Training and Validation Metrics'

    # 确定子图数量
    if learning_rates is not None and len(learning_rates) > 0:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Set overall title
    fig.suptitle(title, fontsize=16, y=0.98)

    # Plot loss curves
    if train_losses and val_losses:
        ax1.plot(train_losses, label='Training Loss', color='#1f77b4', linewidth=2)
        ax1.plot(val_losses, label='Validation Loss', color='#ff7f0e', linewidth=2)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Training and Validation Loss', fontsize=14)
        ax1.legend(fontsize=10)
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # Mark best validation loss point
        if val_losses:
            best_val_epoch = np.argmin(val_losses)
            ax1.axvline(x=best_val_epoch, color='r', linestyle='--', alpha=0.5, label=f'Best Val Epoch: {best_val_epoch}')

    # Plot accuracy curves
    if train_accuracies and val_accuracies:
        ax2.plot(train_accuracies, label='Training Accuracy', color='#1f77b4', linewidth=2)
        ax2.plot(val_accuracies, label='Validation Accuracy', color='#ff7f0e', linewidth=2)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Accuracy', fontsize=12)
        ax2.set_title('Training and Validation Accuracy', fontsize=14)
        ax2.legend(fontsize=10)
        ax2.grid(True, linestyle='--', alpha=0.7)
        
        # Mark best validation accuracy point
        if val_accuracies:
            best_acc_epoch = np.argmax(val_accuracies)
            ax2.axvline(x=best_acc_epoch, color='r', linestyle='--', alpha=0.5, label=f'Best Acc Epoch: {best_acc_epoch}')
    
    # Plot learning rate curve if available
    if learning_rates and len(learning_rates) > 0:
        ax3.plot(learning_rates, label='Learning Rate', color='#2ca02c', linewidth=2)
        ax3.set_xlabel('Epoch', fontsize=12)
        ax3.set_ylabel('Learning Rate', fontsize=12)
        ax3.set_title('Learning Rate Schedule', fontsize=14)
        ax3.legend(fontsize=10)
        ax3.grid(True, linestyle='--', alpha=0.7)
        ax3.set_yscale('log')  # 使用对数刻度显示学习率
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"训练历史图表已保存到: {save_path}")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Training history plot saved to {save_path}")


def plot_confusion_matrix(y_true, y_pred, class_names, save_path='results/confusion_matrix.png',
                         figsize=(8, 6), normalize=False):
    """绘制混淆矩阵"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        title = 'Normalized Confusion Matrix'
    else:
        fmt = 'd'
        title = 'Confusion Matrix'
    
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title, fontsize=14)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Confusion matrix saved to {save_path}")


def plot_audio_waveform(audio_data, sr, save_path='results/waveform.png', 
                        figsize=(12, 4), title='Audio Waveform'):
    """绘制音频波形"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.figure(figsize=figsize)
    time_axis = np.linspace(0, len(audio_data) / sr, len(audio_data))
    plt.plot(time_axis, audio_data, linewidth=0.5)
    plt.title(title, fontsize=14)
    plt.xlabel('Time (seconds)', fontsize=12)
    plt.ylabel('Amplitude', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Waveform plot saved to {save_path}")


def plot_spectrogram(audio_data, sr, save_path='results/spectrogram.png',
                    figsize=(12, 6), title='Spectrogram'):
    """绘制频谱图"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.figure(figsize=figsize)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(audio_data)), ref=np.max)
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title, fontsize=14)
    plt.xlabel('Time (seconds)', fontsize=12)
    plt.ylabel('Frequency (Hz)', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Spectrogram saved to {save_path}")


def plot_mel_spectrogram(audio_data, sr, save_path='results/mel_spectrogram.png',
                        figsize=(12, 6), title='Mel Spectrogram', n_mels=128):
    """绘制梅尔频谱图"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.figure(figsize=figsize)
    mel_spec = librosa.feature.melspectrogram(y=audio_data, sr=sr, n_mels=n_mels)
    mel_spec_db = librosa.amplitude_to_db(mel_spec, ref=np.max)
    
    librosa.display.specshow(mel_spec_db, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title, fontsize=14)
    plt.xlabel('Time (seconds)', fontsize=12)
    plt.ylabel('Mel Frequency', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Mel spectrogram saved to {save_path}")


def plot_feature_distribution(features, labels, feature_names, save_path='results/feature_distribution.png',
                             figsize=(15, 10)):
    """绘制特征分布图"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    n_features = len(feature_names)
    n_cols = 4
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_rows > 1 else [axes]
    
    unique_labels = np.unique(labels)
    colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))
    
    for i, feature_name in enumerate(feature_names):
        ax = axes[i]
        
        for j, label in enumerate(unique_labels):
            mask = labels == label
            ax.hist(features[mask, i], alpha=0.7, label=f'Class {label}', 
                   color=colors[j], bins=30)
        
        ax.set_title(feature_name, fontsize=10)
        ax.set_xlabel('Value', fontsize=8)
        ax.set_ylabel('Frequency', fontsize=8)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(n_features, len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle('Feature Distribution by Class', fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Feature distribution plot saved to {save_path}")


def plot_model_performance(metrics_dict, save_path='results/model_performance.png',
                          figsize=(12, 8)):
    """绘制模型性能对比图"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    models = list(metrics_dict.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        values = [metrics_dict[model].get(metric, 0) for model in models]
        
        bars = ax.bar(models, values, color=plt.cm.viridis(np.linspace(0, 1, len(models))))
        ax.set_title(f'{metric.capitalize()}', fontsize=14)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value:.3f}', ha='center', va='bottom', fontsize=10)
        
        ax.grid(True, alpha=0.3)
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    plt.suptitle('Model Performance Comparison', fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Model performance plot saved to {save_path}")


def plot_audio_features(audio_path_or_data, preprocessor_or_sr=None, save_path='results/audio_features.png',
                       figsize=(15, 10), title='Audio Features Analysis', show_mfcc=True, show_chroma=False):
    """绘制音频特征分析图（包含波形、频谱图、梅尔频谱图等）
    
    Args:
        audio_path_or_data: 音频文件路径或音频数据数组
        preprocessor_or_sr: AudioPreprocessor对象或采样率
        save_path: 保存路径
        figsize: 图像大小
        title: 图像标题
        show_mfcc: 是否显示MFCC特征
        show_chroma: 是否显示色度特征
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 处理输入参数
    if isinstance(audio_path_or_data, str):
        # 如果是文件路径，使用preprocessor加载音频
        if preprocessor_or_sr is None:
            raise ValueError("当传入音频路径时，必须提供preprocessor")
        audio_data = preprocessor_or_sr.load_audio(audio_path_or_data)
        sr = preprocessor_or_sr.sample_rate
    else:
        # 如果是音频数据，直接使用
        audio_data = audio_path_or_data
        sr = preprocessor_or_sr if preprocessor_or_sr is not None else 22050
    
    if audio_data is None:
        logger.error("无法加载音频数据")
        return
    
    # 根据参数决定子图布局
    if show_mfcc and show_chroma:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        subplot_positions = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
    elif show_mfcc or show_chroma:
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        subplot_positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
    else:
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        subplot_positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
    
    fig.suptitle(title, fontsize=16)
    
    plot_idx = 0
    
    # 1. 波形图
    if len(subplot_positions) > plot_idx:
        ax = axes[subplot_positions[plot_idx]] if len(subplot_positions) > 4 else axes[subplot_positions[plot_idx][0], subplot_positions[plot_idx][1]]
        time_axis = np.linspace(0, len(audio_data) / sr, len(audio_data))
        ax.plot(time_axis, audio_data, linewidth=0.5, color='blue')
        ax.set_title('Waveform', fontsize=12)
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Amplitude')
        ax.grid(True, alpha=0.3)
        plot_idx += 1
    
    # 2. 频谱图
    if len(subplot_positions) > plot_idx:
        ax = axes[subplot_positions[plot_idx]] if len(subplot_positions) > 4 else axes[subplot_positions[plot_idx][0], subplot_positions[plot_idx][1]]
        D = librosa.amplitude_to_db(np.abs(librosa.stft(audio_data)), ref=np.max)
        im = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz', ax=ax)
        ax.set_title('Spectrogram', fontsize=12)
        fig.colorbar(im, ax=ax, format='%+2.0f dB')
        plot_idx += 1
    
    # 3. 梅尔频谱图
    if len(subplot_positions) > plot_idx:
        ax = axes[subplot_positions[plot_idx]] if len(subplot_positions) > 4 else axes[subplot_positions[plot_idx][0], subplot_positions[plot_idx][1]]
        mel_spec = librosa.feature.melspectrogram(y=audio_data, sr=sr, n_mels=128)
        mel_spec_db = librosa.amplitude_to_db(mel_spec, ref=np.max)
        im = librosa.display.specshow(mel_spec_db, sr=sr, x_axis='time', y_axis='mel', ax=ax)
        ax.set_title('Mel Spectrogram', fontsize=12)
        fig.colorbar(im, ax=ax, format='%+2.0f dB')
        plot_idx += 1
    
    # 4. MFCC特征（可选）
    if show_mfcc and len(subplot_positions) > plot_idx:
        ax = axes[subplot_positions[plot_idx]] if len(subplot_positions) > 4 else axes[subplot_positions[plot_idx][0], subplot_positions[plot_idx][1]]
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13)
        im = librosa.display.specshow(mfccs, sr=sr, x_axis='time', ax=ax)
        ax.set_title('MFCC Features', fontsize=12)
        ax.set_ylabel('MFCC Coefficients')
        fig.colorbar(im, ax=ax)
        plot_idx += 1
    
    # 5. 色度特征（可选）
    if show_chroma and len(subplot_positions) > plot_idx:
        ax = axes[subplot_positions[plot_idx]] if len(subplot_positions) > 4 else axes[subplot_positions[plot_idx][0], subplot_positions[plot_idx][1]]
        chroma = librosa.feature.chroma_stft(y=audio_data, sr=sr)
        im = librosa.display.specshow(chroma, sr=sr, x_axis='time', y_axis='chroma', ax=ax)
        ax.set_title('Chroma Features', fontsize=12)
        ax.set_ylabel('Chroma')
        fig.colorbar(im, ax=ax)
        plot_idx += 1
    
    # 隐藏未使用的子图
    if len(subplot_positions) > 4:  # 2x3布局
        for i in range(plot_idx, 6):
            if i < len(subplot_positions):
                axes[subplot_positions[i]].set_visible(False)
    else:  # 2x2布局
        for i in range(plot_idx, 4):
            if i < len(subplot_positions):
                axes[subplot_positions[i][0], subplot_positions[i][1]].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Audio features plot saved to {save_path}")