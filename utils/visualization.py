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

# 设置matplotlib字体，支持中文显示
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans", "Arial", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False  # Fix minus sign display
plt.rcParams["font.family"] = "sans-serif"


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


def generate_audio_visualizations(audio_path, preprocessor, save_dir='results/audio_analysis'):
    """生成音频特征可视化（从individual_visualization.py整合）
    
    Args:
        audio_path: 音频文件路径
        preprocessor: 音频预处理器
        save_dir: 保存目录
    """
    os.makedirs(save_dir, exist_ok=True)
    
    try:
        # 加载音频数据
        audio_data = preprocessor.load_audio(audio_path)
        if audio_data is None:
            logger.error(f"无法加载音频文件: {audio_path}")
            return
        
        sr = preprocessor.sample_rate
        
        # 生成各种可视化
        plot_audio_waveform(audio_data, sr, 
                           save_path=os.path.join(save_dir, 'waveform.png'),
                           title=f'Audio Waveform - {os.path.basename(audio_path)}')
        
        plot_spectrogram(audio_data, sr,
                        save_path=os.path.join(save_dir, 'spectrogram.png'),
                        title=f'Spectrogram - {os.path.basename(audio_path)}')
        
        plot_mel_spectrogram(audio_data, sr,
                            save_path=os.path.join(save_dir, 'mel_spectrogram.png'),
                            title=f'Mel Spectrogram - {os.path.basename(audio_path)}')
        
        # 综合特征分析
        plot_audio_features(audio_data, sr,
                           save_path=os.path.join(save_dir, 'audio_features.png'),
                           title=f'Audio Features Analysis - {os.path.basename(audio_path)}',
                           show_mfcc=True, show_chroma=True)
        
        logger.info(f"音频可视化已保存到: {save_dir}")
        
    except Exception as e:
        logger.error(f"生成音频可视化时出错: {str(e)}")


def plot_single_inference_result(predicted_label, confidence, all_probabilities, class_names, 
                                audio_filename, save_path='results/single_inference.png', 
                                figsize=(14, 10)):
    """绘制单个推理结果的直观可视化
    
    Args:
        predicted_label: 预测的标签（字符串）
        confidence: 预测置信度
        all_probabilities: 所有类别的概率分布
        class_names: 类别名称列表
        audio_filename: 音频文件名
        save_path: 保存路径
        figsize: 图像大小
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 2, 1], hspace=0.3, wspace=0.3)
    
    # 主标题
    fig.suptitle(f'车辆个体识别结果 - {audio_filename}', fontsize=18, fontweight='bold')
    
    # 1. 预测结果展示（大标题区域）
    ax_result = fig.add_subplot(gs[0, :])
    ax_result.text(0.5, 0.7, f'预测结果: {predicted_label}', 
                   ha='center', va='center', fontsize=24, fontweight='bold', 
                   color='darkblue', transform=ax_result.transAxes)
    ax_result.text(0.5, 0.3, f'置信度: {confidence:.4f} ({confidence*100:.2f}%)', 
                   ha='center', va='center', fontsize=18, 
                   color='darkgreen' if confidence > 0.5 else 'darkorange',
                   transform=ax_result.transAxes)
    ax_result.set_xlim(0, 1)
    ax_result.set_ylim(0, 1)
    ax_result.axis('off')
    
    # 2. Top-5 预测概率（左侧）
    ax_top5 = fig.add_subplot(gs[1, 0])
    top5_indices = np.argsort(all_probabilities)[-5:][::-1]
    top5_probs = all_probabilities[top5_indices]
    top5_labels = [class_names[i] for i in top5_indices]
    
    colors = ['#2E8B57' if i == 0 else '#4682B4' if i == 1 else '#708090' for i in range(5)]
    bars = ax_top5.barh(range(5), top5_probs, color=colors)
    ax_top5.set_yticks(range(5))
    ax_top5.set_yticklabels(top5_labels)
    ax_top5.set_xlabel('预测概率')
    ax_top5.set_title('Top-5 预测结果', fontsize=14, fontweight='bold')
    ax_top5.grid(axis='x', alpha=0.3)
    
    # 在柱状图上添加数值标签
    for i, (bar, prob) in enumerate(zip(bars, top5_probs)):
        ax_top5.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                     f'{prob:.4f}', va='center', fontsize=10)
    
    # 3. 车辆类型分析（右侧）
    ax_vehicle = fig.add_subplot(gs[1, 1])
    
    # 按车辆类型分组概率
    vehicle_types = ['sedan', 'suv', 'truck', 'motorcycle', 'bus']
    type_probs = {vtype: 0 for vtype in vehicle_types}
    
    for i, prob in enumerate(all_probabilities):
        class_name = class_names[i]
        for vtype in vehicle_types:
            if vtype in class_name:
                type_probs[vtype] += prob
                break
    
    type_names = list(type_probs.keys())
    type_values = list(type_probs.values())
    type_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
    
    # 只显示非零的类型
    filtered_types = [(name, value, color) for name, value, color in zip(type_names, type_values, type_colors) if value > 0.001]
    if filtered_types:
        f_names, f_values, f_colors = zip(*filtered_types)
        wedges, texts, autotexts = ax_vehicle.pie(f_values, labels=f_names, 
                                                  colors=f_colors, autopct='%1.2f%%',
                                                  startangle=90, 
                                                  textprops={'fontsize': 10},
                                                  pctdistance=0.85,
                                                  labeldistance=1.1)
        # 调整标签字体大小
        for text in texts:
            text.set_fontsize(9)
        for autotext in autotexts:
            autotext.set_fontsize(8)
            autotext.set_color('white')
            autotext.set_fontweight('bold')
    else:
        ax_vehicle.text(0.5, 0.5, '无数据', ha='center', va='center', transform=ax_vehicle.transAxes)
    
    ax_vehicle.set_title('车辆类型概率分布', fontsize=14, fontweight='bold')
    
    # 4. 置信度评估（底部）
    ax_confidence = fig.add_subplot(gs[2, :])
    
    # 置信度条形图
    conf_colors = ['red' if confidence < 0.3 else 'orange' if confidence < 0.7 else 'green']
    ax_confidence.barh([0], [confidence], color=conf_colors, height=0.3)
    ax_confidence.set_xlim(0, 1)
    ax_confidence.set_ylim(-0.5, 0.5)
    ax_confidence.set_xlabel('置信度')
    ax_confidence.set_yticks([])
    
    # 添加置信度区间标记
    ax_confidence.axvline(0.3, color='red', linestyle='--', alpha=0.7, label='低置信度')
    ax_confidence.axvline(0.7, color='orange', linestyle='--', alpha=0.7, label='中等置信度')
    ax_confidence.text(0.15, 0, '低', ha='center', va='center', fontweight='bold', color='red')
    ax_confidence.text(0.5, 0, '中', ha='center', va='center', fontweight='bold', color='orange')
    ax_confidence.text(0.85, 0, '高', ha='center', va='center', fontweight='bold', color='green')
    
    # 添加当前置信度标记
    ax_confidence.text(confidence, 0.2, f'{confidence:.4f}', ha='center', va='bottom', 
                       fontweight='bold', fontsize=12)
    
    ax_confidence.set_title('置信度评估', fontsize=14, fontweight='bold')
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Single inference result plot saved to {save_path}")


def plot_inference_results(predictions, true_labels, class_names, confidence_scores=None,
                          save_path='results/inference_results.png', figsize=(12, 8)):
    """绘制推理结果（从individual_visualization.py整合）
    
    Args:
        predictions: 预测结果列表
        true_labels: 真实标签列表
        class_names: 类别名称列表
        confidence_scores: 置信度分数列表（可选）
        save_path: 保存路径
        figsize: 图像大小
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle('Inference Results Analysis', fontsize=16)
    
    # 1. 混淆矩阵
    cm = confusion_matrix(true_labels, predictions)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names, ax=axes[0, 0])
    axes[0, 0].set_title('Confusion Matrix')
    axes[0, 0].set_xlabel('Predicted')
    axes[0, 0].set_ylabel('True')
    
    # 2. 预测分布
    unique_preds, pred_counts = np.unique(predictions, return_counts=True)
    axes[0, 1].bar([class_names[i] for i in unique_preds], pred_counts, color='skyblue')
    axes[0, 1].set_title('Prediction Distribution')
    axes[0, 1].set_xlabel('Predicted Class')
    axes[0, 1].set_ylabel('Count')
    plt.setp(axes[0, 1].get_xticklabels(), rotation=45)
    
    # 3. 真实标签分布
    unique_true, true_counts = np.unique(true_labels, return_counts=True)
    axes[1, 0].bar([class_names[i] for i in unique_true], true_counts, color='lightcoral')
    axes[1, 0].set_title('True Label Distribution')
    axes[1, 0].set_xlabel('True Class')
    axes[1, 0].set_ylabel('Count')
    plt.setp(axes[1, 0].get_xticklabels(), rotation=45)
    
    # 4. 置信度分布（如果提供）
    if confidence_scores is not None:
        axes[1, 1].hist(confidence_scores, bins=20, color='lightgreen', alpha=0.7)
        axes[1, 1].set_title('Confidence Score Distribution')
        axes[1, 1].set_xlabel('Confidence Score')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].axvline(np.mean(confidence_scores), color='red', linestyle='--', 
                          label=f'Mean: {np.mean(confidence_scores):.3f}')
        axes[1, 1].legend()
    else:
        # 如果没有置信度，显示准确率
        accuracy = np.mean(np.array(predictions) == np.array(true_labels))
        axes[1, 1].text(0.5, 0.5, f'Accuracy\n{accuracy:.3f}', 
                        ha='center', va='center', fontsize=24, 
                        transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Overall Accuracy')
        axes[1, 1].set_xticks([])
        axes[1, 1].set_yticks([])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Inference results plot saved to {save_path}")


def generate_prediction_visualizations(predictions, probabilities, class_names, 
                                     save_dir='results/predictions', top_k=5):
    """生成预测结果可视化（从individual_visualization.py整合）
    
    Args:
        predictions: 预测结果列表
        probabilities: 预测概率矩阵
        class_names: 类别名称列表
        save_dir: 保存目录
        top_k: 显示前k个预测结果
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. 预测概率分布
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Prediction Analysis', fontsize=16)
    
    # 平均预测概率
    mean_probs = np.mean(probabilities, axis=0)
    axes[0, 0].bar(class_names, mean_probs, color='skyblue')
    axes[0, 0].set_title('Average Prediction Probabilities')
    axes[0, 0].set_xlabel('Class')
    axes[0, 0].set_ylabel('Average Probability')
    plt.setp(axes[0, 0].get_xticklabels(), rotation=45)
    
    # 预测置信度分布
    max_probs = np.max(probabilities, axis=1)
    axes[0, 1].hist(max_probs, bins=20, color='lightgreen', alpha=0.7)
    axes[0, 1].set_title('Prediction Confidence Distribution')
    axes[0, 1].set_xlabel('Max Probability')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].axvline(np.mean(max_probs), color='red', linestyle='--',
                      label=f'Mean: {np.mean(max_probs):.3f}')
    axes[0, 1].legend()
    
    # 类别预测频次
    unique_preds, pred_counts = np.unique(predictions, return_counts=True)
    axes[1, 0].bar([class_names[i] for i in unique_preds], pred_counts, color='lightcoral')
    axes[1, 0].set_title('Prediction Frequency')
    axes[1, 0].set_xlabel('Predicted Class')
    axes[1, 0].set_ylabel('Count')
    plt.setp(axes[1, 0].get_xticklabels(), rotation=45)
    
    # Top-K预测准确性
    if len(probabilities) > 0:
        top_k_indices = np.argsort(probabilities, axis=1)[:, -top_k:]
        top_k_probs = np.sort(probabilities, axis=1)[:, -top_k:]
        
        # 显示前几个样本的top-k预测
        sample_indices = np.random.choice(len(predictions), min(5, len(predictions)), replace=False)
        
        axes[1, 1].set_title(f'Top-{top_k} Predictions (Sample)')
        y_pos = np.arange(len(sample_indices))
        
        for i, sample_idx in enumerate(sample_indices):
            top_classes = [class_names[idx] for idx in top_k_indices[sample_idx]]
            top_probs = top_k_probs[sample_idx]
            
            # 只显示最高的预测
            axes[1, 1].barh(i, top_probs[-1], color='gold')
            axes[1, 1].text(top_probs[-1]/2, i, f'{top_classes[-1]}\n{top_probs[-1]:.3f}',
                           ha='center', va='center', fontsize=8)
        
        axes[1, 1].set_yticks(y_pos)
        axes[1, 1].set_yticklabels([f'Sample {i+1}' for i in range(len(sample_indices))])
        axes[1, 1].set_xlabel('Probability')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'prediction_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Prediction visualizations saved to {save_dir}")


def generate_summary_visualization(training_history, test_results, model_info,
                                 save_path='results/training_summary.png', figsize=(16, 10)):
    """生成训练总结可视化（从individual_visualization.py整合）
    
    Args:
        training_history: 训练历史字典
        test_results: 测试结果字典
        model_info: 模型信息字典
        save_path: 保存路径
        figsize: 图像大小
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. 训练损失曲线
    ax1 = fig.add_subplot(gs[0, 0])
    if 'train_loss' in training_history and 'val_loss' in training_history:
        ax1.plot(training_history['train_loss'], label='Train Loss', color='blue')
        ax1.plot(training_history['val_loss'], label='Val Loss', color='orange')
        ax1.set_title('Training Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # 2. 训练准确率曲线
    ax2 = fig.add_subplot(gs[0, 1])
    if 'train_acc' in training_history and 'val_acc' in training_history:
        ax2.plot(training_history['train_acc'], label='Train Acc', color='blue')
        ax2.plot(training_history['val_acc'], label='Val Acc', color='orange')
        ax2.set_title('Training Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # 3. 学习率曲线
    ax3 = fig.add_subplot(gs[0, 2])
    if 'learning_rates' in training_history:
        ax3.plot(training_history['learning_rates'], color='green')
        ax3.set_title('Learning Rate')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Learning Rate')
        ax3.set_yscale('log')
        ax3.grid(True, alpha=0.3)
    
    # 4. 测试结果指标
    ax4 = fig.add_subplot(gs[1, :])
    if test_results:
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        values = [test_results.get(metric, 0) for metric in metrics]
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        bars = ax4.bar(metrics, values, color=colors, alpha=0.7)
        ax4.set_title('Test Results', fontsize=14)
        ax4.set_ylabel('Score')
        ax4.set_ylim(0, 1)
        
        # 添加数值标签
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        ax4.grid(True, alpha=0.3)
    
    # 5. 模型信息文本
    ax5 = fig.add_subplot(gs[2, :])
    ax5.axis('off')
    
    info_text = "Model Information:\n"
    if model_info:
        for key, value in model_info.items():
            info_text += f"{key}: {value}\n"
    
    # 添加训练总结
    if training_history:
        best_val_acc = max(training_history.get('val_acc', [0]))
        best_val_loss = min(training_history.get('val_loss', [float('inf')]))
        total_epochs = len(training_history.get('train_loss', []))
        
        info_text += f"\nTraining Summary:\n"
        info_text += f"Total Epochs: {total_epochs}\n"
        info_text += f"Best Validation Accuracy: {best_val_acc:.4f}\n"
        info_text += f"Best Validation Loss: {best_val_loss:.4f}\n"
    
    ax5.text(0.05, 0.95, info_text, transform=ax5.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
    
    plt.suptitle('Training Summary Report', fontsize=18, y=0.98)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Training summary visualization saved to {save_path}")