#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
音频特征提取和识别结果可视化模块
专门用于单一音频文件的特征可视化和识别结果展示
"""

# 使用公共工具模块统一导入
from core.common_utils import (
    os, sys, time, json, Path,
    List, Dict, Optional, Tuple, Any, Union,
    dataclass, asdict, warnings, np,
    config_manager, file_manager, logger_factory, TimerContext
)

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import librosa
import librosa.display
import seaborn as sns
from datetime import datetime

# 初始化日志器
visualizer_logger = logger_factory.get_logger("音频可视化")

# 设置字体和样式
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

class AudioVisualizer:
    """音频特征和识别结果可视化器"""
    
    def __init__(self, figsize: Tuple[int, int] = (15, 10), dpi: int = 100):
        """
        初始化可视化器
        
        Args:
            figsize: 图像大小
            dpi: 图像分辨率
        """
        self.figsize = figsize
        self.dpi = dpi
        self.colors = {
            'primary': '#2E86AB',
            'secondary': '#A23B72', 
            'accent': '#F18F01',
            'success': '#C73E1D',
            'background': '#F5F5F5'
        }
    
    def _format_vehicle_name(self, vehicle_id: str) -> str:
        """简化车辆名称格式化 - 去除复杂解析逻辑"""
        if '_' not in vehicle_id:
            return vehicle_id.title()
        
        parts = vehicle_id.split('_')
        vehicle_type = parts[0].title()
        
        # 提取数字部分
        numbers = [part for part in parts[1:] if part.isdigit()]
        if numbers:
            return f"{vehicle_type} {numbers[0]}"
        
        return vehicle_type
    
    def load_audio_for_visualization(self, audio_path: str, sr: int = 16000) -> Tuple[Optional[np.ndarray], Optional[int]]:
        """
        加载音频文件用于可视化
        
        Args:
            audio_path: 音频文件路径
            sr: 采样率
            
        Returns:
            音频数据和采样率
        """
        try:
            y, sr = librosa.load(audio_path, sr=sr)
            return y, sr
        except Exception as e:
            print(f"音频加载失败: {e}")
            return None, None
    
    def visualize_audio_features(self, audio_path: str, save_path: Optional[str] = None) -> str:
        """
        可视化音频特征
        
        Args:
            audio_path: 音频文件路径
            save_path: 保存路径，如果为None则自动生成
            
        Returns:
            保存的图像文件路径
        """
        # 加载音频
        y, sr = self.load_audio_for_visualization(audio_path)
        if y is None:
            raise ValueError(f"无法加载音频文件: {audio_path}")
        
        # 创建图像
        fig = plt.figure(figsize=self.figsize, dpi=self.dpi)
        gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)
        
        # 1. 波形图
        ax1 = fig.add_subplot(gs[0, :])
        time_axis = np.linspace(0, len(y) / sr, len(y))
        ax1.plot(time_axis, y, color=self.colors['primary'], linewidth=0.8)
        ax1.set_title('Audio Waveform', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Time (seconds)')
        ax1.set_ylabel('Amplitude')
        ax1.grid(True, alpha=0.3)
        
        # 2. 频谱图
        ax2 = fig.add_subplot(gs[1, 0])
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        img = librosa.display.specshow(D, y_axis='hz', x_axis='time', sr=sr, ax=ax2, cmap='viridis')
        ax2.set_title('Spectrogram', fontsize=12, fontweight='bold')
        plt.colorbar(img, ax=ax2, format='%+2.0f dB')
        
        # 3. Mel频谱图
        ax3 = fig.add_subplot(gs[1, 1])
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        mel_spec_db = librosa.amplitude_to_db(mel_spec, ref=np.max)
        img2 = librosa.display.specshow(mel_spec_db, y_axis='mel', x_axis='time', sr=sr, ax=ax3, cmap='plasma')
        ax3.set_title('Mel Spectrogram', fontsize=12, fontweight='bold')
        plt.colorbar(img2, ax=ax3, format='%+2.0f dB')
        
        # 4. MFCC特征
        ax4 = fig.add_subplot(gs[2, 0])
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        img3 = librosa.display.specshow(mfccs, x_axis='time', ax=ax4, cmap='coolwarm')
        ax4.set_title('MFCC Features', fontsize=12, fontweight='bold')
        plt.colorbar(img3, ax=ax4)
        
        # 5. 频谱质心和带宽
        ax5 = fig.add_subplot(gs[2, 1])
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        
        frames = range(len(spectral_centroids))
        t = librosa.frames_to_time(frames, sr=sr)
        
        ax5.plot(t, spectral_centroids, color=self.colors['accent'], label='Spectral Centroid', linewidth=2)
        ax5_twin = ax5.twinx()
        ax5_twin.plot(t, spectral_bandwidth, color=self.colors['secondary'], label='Spectral Bandwidth', linewidth=2)
        
        ax5.set_title('Spectral Features', fontsize=12, fontweight='bold')
        ax5.set_xlabel('Time (seconds)')
        ax5.set_ylabel('Spectral Centroid (Hz)', color=self.colors['accent'])
        ax5_twin.set_ylabel('Spectral Bandwidth (Hz)', color=self.colors['secondary'])
        ax5.legend(loc='upper left')
        ax5_twin.legend(loc='upper right')
        
        # 添加总标题
        audio_name = Path(audio_path).name
        fig.suptitle(f'Audio Feature Analysis - {audio_name}', fontsize=16, fontweight='bold')
        
        # 保存图像
        if save_path is None:
            from config import PATH_CONFIG
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f"{PATH_CONFIG['results_dir']}/audio_features_{Path(audio_path).stem}_{timestamp}.png"
        
        os.makedirs(Path(save_path).parent, exist_ok=True)
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return save_path
    
    def visualize_recognition_result(self, result_data: Dict[str, Any], audio_info: Dict[str, Any], 
                                   save_path: Optional[str] = None) -> str:
        """
        Visualize recognition results with modern and intuitive design
        
        Args:
            result_data: Recognition result data
            audio_info: Audio information
            save_path: Save path
            
        Returns:
            str: Path to saved image
        """
        # Create figure with modern layout
        fig = plt.figure(figsize=(16, 10), facecolor='white')
        gs = GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.3, 
                     height_ratios=[1, 2, 2], width_ratios=[1, 1, 1])
        
        # Extract data
        vehicle_id = result_data.get('vehicle_id', 'Unknown')
        confidence = result_data.get('confidence', 0.0)
        method = result_data.get('method', 'Unknown')
        processing_time = result_data.get('processing_time', 0.0)
        
        audio_name = audio_info.get('file_name', 'Unknown')
        file_size = audio_info.get('file_size', 0)
        duration = audio_info.get('duration', 0.0)
        sample_rate = audio_info.get('sample_rate', 0)
        
        # 1. Header with main result
        ax_header = fig.add_subplot(gs[0, :])
        ax_header.axis('off')
        
        # Main result display
        result_color = self.colors['success'] if confidence >= 0.8 else \
                      self.colors['accent'] if confidence >= 0.6 else self.colors['secondary']
        
        ax_header.text(0.5, 0.7, f"Vehicle Identified: {vehicle_id}", 
                      ha='center', va='center', fontsize=20, fontweight='bold',
                      transform=ax_header.transAxes, color=result_color)
        
        ax_header.text(0.5, 0.3, f"Confidence: {confidence:.1%} | Processing Time: {processing_time:.2f}s", 
                      ha='center', va='center', fontsize=14,
                      transform=ax_header.transAxes, color='gray')
        
        # 2. Confidence gauge (clean circular design)
        ax_gauge = fig.add_subplot(gs[1, 0])
        
        # Background circle (full circle)
        circle_bg = plt.Circle((0, 0), 0.8, fill=False, linewidth=12, color='lightgray', alpha=0.3)
        ax_gauge.add_patch(circle_bg)
        
        # Confidence arc (clean single arc)
        conf_angle = confidence * 2 * np.pi
        theta = np.linspace(-np.pi/2, -np.pi/2 + conf_angle, 100)
        
        # Draw confidence arc as a single thick line
        x_arc = 0.8 * np.cos(theta)
        y_arc = 0.8 * np.sin(theta)
        ax_gauge.plot(x_arc, y_arc, color=result_color, linewidth=12, alpha=0.8, solid_capstyle='round')
        
        # Center percentage text (larger and clearer)
        ax_gauge.text(0, 0.1, f'{confidence:.0%}', ha='center', va='center', 
                     fontsize=28, fontweight='bold', color=result_color)
        ax_gauge.text(0, -0.25, 'Confidence', ha='center', va='center', 
                     fontsize=14, color='gray', fontweight='bold')
        
        # Add percentage symbol separately for better visibility
        ax_gauge.text(0, -0.45, f'({confidence:.1%})', ha='center', va='center', 
                     fontsize=12, color='gray', style='italic')
        
        ax_gauge.set_xlim(-1.2, 1.2)
        ax_gauge.set_ylim(-1.2, 1.2)
        ax_gauge.set_aspect('equal')
        ax_gauge.axis('off')
        ax_gauge.set_title('Confidence Assessment', fontsize=14, fontweight='bold', pad=20)
        
        # 3. Audio information panel
        ax_info = fig.add_subplot(gs[1, 1])
        ax_info.axis('off')
        
        info_items = [
            ('Audio File', audio_name),
            ('File Size', f'{file_size / 1024:.1f} KB' if file_size > 0 else 'Unknown'),
            ('Duration', f'{duration:.2f}s'),
            ('Sample Rate', f'{sample_rate} Hz'),
            ('Method', method.upper())
        ]
        
        y_positions = np.linspace(0.9, 0.1, len(info_items))
        for i, (label, value) in enumerate(info_items):
            ax_info.text(0.05, y_positions[i], f'{label}:', fontweight='bold', 
                        transform=ax_info.transAxes, fontsize=11)
            ax_info.text(0.55, y_positions[i], str(value), 
                        transform=ax_info.transAxes, fontsize=11, color='navy')
        
        # Add border
        rect = patches.Rectangle((0.02, 0.05), 0.96, 0.9, linewidth=2, 
                               edgecolor='lightblue', facecolor='lightblue', 
                               alpha=0.1, transform=ax_info.transAxes)
        ax_info.add_patch(rect)
        ax_info.set_title('Audio Information', fontsize=14, fontweight='bold', pad=20)
        
        # 4. Performance indicator
        ax_perf = fig.add_subplot(gs[1, 2])
        ax_perf.axis('off')
        
        # Performance metrics
        perf_score = "Excellent" if confidence >= 0.9 else \
                    "Good" if confidence >= 0.7 else \
                    "Fair" if confidence >= 0.5 else "Poor"
        
        perf_color = self.colors['success'] if confidence >= 0.9 else \
                    self.colors['accent'] if confidence >= 0.7 else \
                    self.colors['accent'] if confidence >= 0.5 else self.colors['secondary']
        
        # Performance indicator circle
        circle_perf = plt.Circle((0.5, 0.6), 0.25, color=perf_color, alpha=0.2, 
                               transform=ax_perf.transAxes)
        ax_perf.add_patch(circle_perf)
        
        ax_perf.text(0.5, 0.6, perf_score, ha='center', va='center', 
                    fontsize=16, fontweight='bold', color=perf_color,
                    transform=ax_perf.transAxes)
        
        ax_perf.text(0.5, 0.3, f'Speed: {processing_time:.2f}s', ha='center', va='center', 
                    fontsize=12, transform=ax_perf.transAxes)
        
        ax_perf.set_title('Performance', fontsize=14, fontweight='bold', pad=20)
        
        # 5. Similarity ranking (clean design without overlapping)
        ax_sim = fig.add_subplot(gs[2, :])
        
        similar_vehicles = result_data.get('similar_vehicles', [])
        if similar_vehicles and len(similar_vehicles) > 0:
            # Take top 5 results
            top_similar = similar_vehicles[:5]
            vehicle_names = []
            similarities = []
            
            for i, item in enumerate(top_similar):
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    original_name = str(item[0])
                    similarity = float(item[1])
                    
                    # 简化的车辆名称解析
                    display_name = self._format_vehicle_name(original_name)
                    
                    vehicle_names.append(display_name)
                    similarities.append(similarity)
            
            if vehicle_names and similarities:
                # Create horizontal bar chart with clean layout
                y_pos = np.arange(len(vehicle_names))
                
                # Use distinct colors for each rank
                colors = ['#2E8B57', '#4682B4', '#DAA520', '#CD853F', '#9370DB'][:len(similarities)]
                
                bars = ax_sim.barh(y_pos, similarities, color=colors, alpha=0.8, height=0.5)
                
                # Add confidence values at the end of bars (no overlap)
                for i, (bar, sim) in enumerate(zip(bars, similarities)):
                    # Place confidence score at the right end of the bar with better styling
                    ax_sim.text(sim + 0.02, bar.get_y() + bar.get_height()/2, 
                              f'{sim:.3f}', va='center', ha='left', fontsize=12, fontweight='bold',
                              color='#2C3E50', bbox=dict(boxstyle="round,pad=0.2", facecolor='white', 
                              edgecolor='#BDC3C7', alpha=0.8))
                
                # 简化的排名标签生成
                custom_labels = [f"#{i+1} {name}" for i, name in enumerate(vehicle_names)]
                
                # Customize appearance with better spacing
                ax_sim.set_yticks(y_pos)
                ax_sim.set_yticklabels(custom_labels, fontsize=11, fontweight='bold')
                ax_sim.set_xlabel('Similarity Score', fontsize=12, fontweight='bold')
                ax_sim.set_title('Top 5 Similar Vehicles', fontsize=14, fontweight='bold', pad=20)
                ax_sim.set_xlim(0, max(similarities) + 0.15)  # More space for confidence values
                ax_sim.grid(True, alpha=0.3, axis='x')
                ax_sim.spines['top'].set_visible(False)
                ax_sim.spines['right'].set_visible(False)
                
                # Removed colored ranking indicators to avoid overlap with vehicle names
                
                # Invert y-axis to show rank 1 at top
                ax_sim.invert_yaxis()
                
            else:
                ax_sim.text(0.5, 0.5, 'No similarity data available', 
                          ha='center', va='center', transform=ax_sim.transAxes, 
                          fontsize=14, style='italic', color='gray')
                ax_sim.axis('off')
        else:
            ax_sim.text(0.5, 0.5, 'No similarity data available', 
                      ha='center', va='center', transform=ax_sim.transAxes, 
                      fontsize=14, style='italic', color='gray')
            ax_sim.axis('off')
        
        # Main title
        fig.suptitle('Vehicle Recognition Analysis Report', 
                    fontsize=18, fontweight='bold', y=0.95)
        
        # Save image
        if save_path is None:
            from config import PATH_CONFIG
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            vehicle_id_clean = result_data.get('vehicle_id', 'unknown').replace('_', '-')
            save_path = f"{PATH_CONFIG['results_dir']}/recognition_result_{vehicle_id_clean}_{timestamp}.png"
        
        os.makedirs(Path(save_path).parent, exist_ok=True)
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return save_path
    
