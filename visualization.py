# utils/visualization.py

import matplotlib.pyplot as plt
import numpy as np
import librosa.display
import os
from sklearn.metrics import classification_report

# 设置matplotlib使用英文字体，避免中文字体缺失警告
plt.rcParams["font.family"] = ["DejaVu Sans", "Arial", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False  # Fix minus sign display


def plot_training_history(train_losses, val_losses, train_accuracies, val_accuracies,
                          save_path='results/training_history.png', figsize=(12, 6),
                          title='Training and Validation Metrics', learning_rates=None):
    """绘制训练历史，支持学习率曲线"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 确定子图数量
    if learning_rates is not None and len(learning_rates) > 0:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Set overall title
    fig.suptitle(title, fontsize=16, y=0.98)

    # Plot loss curves
    ax1.plot(train_losses, label='Training Loss', color='#1f77b4', linewidth=2)
    ax1.plot(val_losses, label='Validation Loss', color='#ff7f0e', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Mark best validation loss point
    best_val_epoch = np.argmin(val_losses)
    ax1.axvline(x=best_val_epoch, color='r', linestyle='--', alpha=0.5, label=f'Best Val Epoch: {best_val_epoch}')

    # Plot accuracy curves
    ax2.plot(train_accuracies, label='Training Accuracy', color='#1f77b4', linewidth=2)
    ax2.plot(val_accuracies, label='Validation Accuracy', color='#ff7f0e', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Training and Validation Accuracy', fontsize=14)
    ax2.set_ylim(0, 105)  # Set y-axis range for better visibility
    ax2.legend(fontsize=10)
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # Mark best validation accuracy point
    best_acc_epoch = np.argmax(val_accuracies)
    ax2.axvline(x=best_acc_epoch, color='r', linestyle='--', alpha=0.5, label=f'Best Val Epoch: {best_acc_epoch}')
    ax2.annotate(f'{val_accuracies[best_acc_epoch]:.2f}%', 
                 xy=(best_acc_epoch, val_accuracies[best_acc_epoch]),
                 xytext=(10, 10), textcoords='offset points',
                 arrowprops=dict(arrowstyle='->'))

    # If learning rates are provided, plot learning rate curve
    if learning_rates is not None and len(learning_rates) > 0:
        ax3.plot(learning_rates, label='Learning Rate', color='#2ca02c', linewidth=2)
        ax3.set_xlabel('Epoch', fontsize=12)
        ax3.set_ylabel('Learning Rate', fontsize=12)
        ax3.set_title('Learning Rate Changes', fontsize=14)
        ax3.set_yscale('log')  # Use log scale to better show learning rate changes
        ax3.legend(fontsize=10)
        ax3.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)  # 调整顶部间距以显示总标题
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(cm, class_names, y_true=None, y_pred=None,
                          save_path='results/confusion_matrix.png', normalize=False,
                          title='Confusion Matrix', figsize=(10, 8), cmap='Blues'):
    """Plot confusion matrix with support for normalization and evaluation metrics"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Normalization option
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        colorbar_label = 'Accuracy'
    else:
        fmt = 'd'
        colorbar_label = 'Count'

    plt.figure(figsize=figsize)
    im = plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=16)
    cbar = plt.colorbar(im)
    cbar.set_label(colorbar_label, fontsize=12)

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha='right', fontsize=11)
    plt.yticks(tick_marks, class_names, fontsize=11)

    # Ensure text color contrast
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], fmt),
                     ha="center", va="center", fontsize=10,
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True Label', fontsize=13)
    plt.xlabel('Predicted Label', fontsize=13)
    
    # 如果提供了真实标签和预测标签，添加分类报告作为文本注释
    if y_true is not None and y_pred is not None:
        report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
        accuracy = report['accuracy']
        
        # 在图表下方添加准确率信息
        plt.figtext(0.5, 0.01, f'总体准确率: {accuracy:.4f}', ha='center', fontsize=12, weight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_audio_features(audio_path, preprocessor, save_path=None, figsize=(12, 10),
                        show_mfcc=False, show_chroma=False):
    """Plot audio features with support for multiple feature visualizations"""
    # Load audio
    audio = preprocessor.load_audio(audio_path)
    audio_name = os.path.basename(audio_path)
    
    # Determine number of subplots
    plots = 2  # At least show waveform and mel spectrogram
    if show_mfcc: plots += 1
    if show_chroma: plots += 1
    
    # Create figure
    fig, axes = plt.subplots(plots, 1, figsize=figsize)
    if plots == 1: axes = [axes]  # Ensure axes is list format
    
    # Set overall title
    fig.suptitle(f'Audio Feature Analysis: {audio_name}', fontsize=16, y=0.98)
    
    # Plot waveform
    ax1 = axes[0]
    librosa.display.waveshow(audio, sr=preprocessor.sample_rate, ax=ax1)
    ax1.set_title('Waveform', fontsize=14)
    ax1.set_xlabel('Time (seconds)', fontsize=12)
    ax1.set_ylabel('Amplitude', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.5)
    # 为3秒音频优化时间轴显示
    ax1.set_xlim(0, 3.0)
    ax1.set_xticks([0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    
    # Plot mel spectrogram
    features = preprocessor.extract_mel_spectrogram(audio)
    ax2 = axes[1]
    img = librosa.display.specshow(features, sr=preprocessor.sample_rate,
                                  hop_length=preprocessor.hop_length,
                                  x_axis='time', y_axis='mel', ax=ax2)
    ax2.set_title('Mel Spectrogram', fontsize=14)
    ax2.set_xlabel('Time (seconds)', fontsize=12)
    ax2.set_ylabel('Mel Frequency', fontsize=12)
    # 为3秒音频优化时间轴显示
    ax2.set_xlim(0, 3.0)
    ax2.set_xticks([0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    fig.colorbar(img, ax=ax2, format='%+2.0f dB', label='Decibels')
    
    # If needed, plot MFCC features
    if show_mfcc:
        try:
            # Try to compute MFCC
            mfcc = librosa.feature.mfcc(y=audio, sr=preprocessor.sample_rate,
                                       n_mfcc=13, hop_length=preprocessor.hop_length)
            ax3 = axes[2]
            img_mfcc = librosa.display.specshow(mfcc, sr=preprocessor.sample_rate,
                                              hop_length=preprocessor.hop_length,
                                              x_axis='time', y_axis='mel', ax=ax3)
            ax3.set_title('MFCC Features', fontsize=14)
            ax3.set_xlabel('Time (seconds)', fontsize=12)
            ax3.set_ylabel('MFCC Coefficients', fontsize=12)
            # 为3秒音频优化时间轴显示
            ax3.set_xlim(0, 3.0)
            ax3.set_xticks([0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
            fig.colorbar(img_mfcc, ax=ax3, label='Coefficient Value')
        except Exception as e:
            print(f"Error computing MFCC: {e}")
    
    # If needed, plot chromagram
    if show_chroma:
        try:
            # Try to compute chroma features
            chroma = librosa.feature.chroma_stft(y=audio, sr=preprocessor.sample_rate,
                                               hop_length=preprocessor.hop_length)
            ax_idx = 3 if show_mfcc else 2
            ax4 = axes[ax_idx]
            img_chroma = librosa.display.specshow(chroma, sr=preprocessor.sample_rate,
                                               hop_length=preprocessor.hop_length,
                                               x_axis='time', y_axis='chroma', ax=ax4)
            ax4.set_title('Chromagram', fontsize=14)
            ax4.set_xlabel('Time (seconds)', fontsize=12)
            ax4.set_ylabel('Pitch Class', fontsize=12)
            # 为3秒音频优化时间轴显示
            ax4.set_xlim(0, 3.0)
            ax4.set_xticks([0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
            fig.colorbar(img_chroma, ax=ax4, label='Intensity')
        except Exception as e:
            print(f"Error computing chroma features: {e}")
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)  # 调整顶部间距以显示总标题
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()


def plot_inference_results(audio_path, predictions, label_mapping, top_k=3, save_path=None, figsize=(10, 6)):
    """Plot inference results as bar chart"""
    # Get class names
    class_names = list(label_mapping.keys())
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Sort prediction results
    sorted_indices = np.argsort(predictions)[::-1][:top_k]  # Take top_k highest confidence
    sorted_classes = [class_names[i] for i in sorted_indices]
    sorted_probs = predictions[sorted_indices]
    
    # Plot bar chart
    colors = plt.cm.RdYlBu_r(np.linspace(0, 1, len(sorted_probs)))
    bars = plt.barh(sorted_classes, sorted_probs, color=colors)
    
    # Add value labels
    for bar, prob in zip(bars, sorted_probs):
        plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f'{prob:.4f}', va='center', fontsize=10)
    
    # Set chart properties
    plt.xlim(0, 1.05)
    plt.xlabel('Confidence', fontsize=12)
    plt.ylabel('Vehicle Class', fontsize=12)
    plt.title(f'Audio Recognition Results: {os.path.basename(audio_path)}', fontsize=14)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()