#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强版可视化模块
- 音频特征合并显示
- 识别结果独立展示
- 优化视觉效果
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import seaborn as sns
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.unified_engine import UnifiedVehicleRecognitionEngine, RecognitionResult, AudioInfo
from enhanced_feature_extractor import EnhancedAudioFeatureExtractor
from core.logger import logger

# 设置英文字体和样式
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
sns.set_palette("husl")


class EnhancedVisualizer:
    """增强版可视化器"""
    
    def __init__(self, output_dir: str = "results"):
        """
        初始化可视化器
        
        Args:
            output_dir: 输出目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # 初始化组件
        self.recognition_engine = UnifiedVehicleRecognitionEngine()
        self.feature_extractor = EnhancedAudioFeatureExtractor()
        
        # 颜色配置
        self.colors = {
            'primary': '#2E86AB',
            'secondary': '#A23B72', 
            'accent': '#F18F01',
            'success': '#C73E1D',
            'background': '#F5F5F5',
            'text': '#2C3E50'
        }
    
    def create_audio_features_plot(self, audio_path: str, audio_info: AudioInfo) -> str:
        """
        创建音频特征综合图
        
        Args:
            audio_path: 音频文件路径
            audio_info: 音频信息
            
        Returns:
            str: 保存的图片路径
        """
        # 加载和处理音频
        import librosa
        y, sr = librosa.load(audio_path, sr=22050)
        
        # 提取基础音频特征用于可视化
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        
        # 提取其他特征
        try:
            chroma = librosa.feature.chroma(y=y, sr=sr)
        except:
            chroma = None
            
        try:
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        except:
            spectral_centroid = None
            
        try:
            zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
        except:
            zero_crossing_rate = None
            
        try:
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        except:
            spectral_rolloff = None
        
        # 创建图形
        fig = plt.figure(figsize=(16, 12))
        fig.suptitle(f'Audio Feature Analysis - {audio_info.file_name}', 
                    fontsize=20, fontweight='bold', color=self.colors['text'])
        
        # 创建网格布局
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # 1. 波形图 (占据第一行)
        ax1 = fig.add_subplot(gs[0, :])
        time_axis = np.linspace(0, len(y) / sr, len(y))
        ax1.plot(time_axis, y, color=self.colors['primary'], linewidth=0.8)
        ax1.set_title('Audio Waveform', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Time (seconds)')
        ax1.set_ylabel('Amplitude')
        ax1.grid(True, alpha=0.3)
        
        # 添加音频信息文本
        info_text = f"File Size: {audio_info.file_size/1024:.1f} KB"
        if audio_info.duration:
            info_text += f" | Duration: {audio_info.duration:.2f}s"
        if audio_info.sample_rate:
            info_text += f" | Sample Rate: {audio_info.sample_rate} Hz"
        ax1.text(0.02, 0.98, info_text, transform=ax1.transAxes, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor=self.colors['background'], alpha=0.8),
                verticalalignment='top', fontsize=10)
        
        # 2. 梅尔频谱图
        ax2 = fig.add_subplot(gs[1, 0])
        im2 = ax2.imshow(mel_spec_db, aspect='auto', origin='lower', 
                        cmap='viridis', interpolation='bilinear')
        ax2.set_title('Mel Spectrogram', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Time Frames')
        ax2.set_ylabel('Mel Frequency')
        plt.colorbar(im2, ax=ax2, shrink=0.8)
        
        # 3. MFCC特征
        ax3 = fig.add_subplot(gs[1, 1])
        im3 = ax3.imshow(mfcc, aspect='auto', origin='lower', 
                        cmap='coolwarm', interpolation='bilinear')
        ax3.set_title('MFCC Features', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Time Frames')
        ax3.set_ylabel('MFCC Coefficients')
        plt.colorbar(im3, ax=ax3, shrink=0.8)
        
        # 4. 色度特征
        ax4 = fig.add_subplot(gs[1, 2])
        if chroma is not None:
            im4 = ax4.imshow(chroma, aspect='auto', origin='lower', 
                            cmap='plasma', interpolation='bilinear')
            ax4.set_title('Chroma Features', fontsize=12, fontweight='bold')
            ax4.set_xlabel('Time Frames')
            ax4.set_ylabel('Chroma')
            plt.colorbar(im4, ax=ax4, shrink=0.8)
        else:
            ax4.text(0.5, 0.5, 'Chroma Features\nNot Available', ha='center', va='center', 
                    transform=ax4.transAxes, fontsize=12)
            ax4.set_title('Chroma Features', fontsize=12, fontweight='bold')
        
        # 5. 频谱质心
        ax5 = fig.add_subplot(gs[2, 0])
        if spectral_centroid is not None:
            centroid = spectral_centroid[0]
            time_frames = np.arange(len(centroid))
            ax5.plot(time_frames, centroid, color=self.colors['accent'], linewidth=2)
            ax5.fill_between(time_frames, centroid, alpha=0.3, color=self.colors['accent'])
            ax5.set_title('Spectral Centroid', fontsize=12, fontweight='bold')
            ax5.set_xlabel('Time Frames')
            ax5.set_ylabel('Frequency (Hz)')
            ax5.grid(True, alpha=0.3)
        else:
            ax5.text(0.5, 0.5, 'Spectral Centroid\nNot Available', ha='center', va='center', 
                    transform=ax5.transAxes, fontsize=12)
            ax5.set_title('Spectral Centroid', fontsize=12, fontweight='bold')
        
        # 6. 零交叉率
        ax6 = fig.add_subplot(gs[2, 1])
        if zero_crossing_rate is not None:
            zcr = zero_crossing_rate[0]
            time_frames = np.arange(len(zcr))
            ax6.plot(time_frames, zcr, color=self.colors['secondary'], linewidth=2)
            ax6.fill_between(time_frames, zcr, alpha=0.3, color=self.colors['secondary'])
            ax6.set_title('Zero Crossing Rate', fontsize=12, fontweight='bold')
            ax6.set_xlabel('Time Frames')
            ax6.set_ylabel('ZCR')
            ax6.grid(True, alpha=0.3)
        else:
            ax6.text(0.5, 0.5, 'Zero Crossing Rate\nNot Available', ha='center', va='center', 
                    transform=ax6.transAxes, fontsize=12)
            ax6.set_title('Zero Crossing Rate', fontsize=12, fontweight='bold')
        
        # 7. 频谱统计
        ax7 = fig.add_subplot(gs[2, 2])
        if spectral_rolloff is not None:
            rolloff = spectral_rolloff[0]
            time_frames = np.arange(len(rolloff))
            ax7.plot(time_frames, rolloff, color=self.colors['success'], linewidth=2)
            ax7.fill_between(time_frames, rolloff, alpha=0.3, color=self.colors['success'])
            ax7.set_title('Spectral Rolloff', fontsize=12, fontweight='bold')
            ax7.set_xlabel('Time Frames')
            ax7.set_ylabel('Frequency (Hz)')
            ax7.grid(True, alpha=0.3)
        else:
            ax7.text(0.5, 0.5, 'Spectral Rolloff\nNot Available', ha='center', va='center', 
                    transform=ax7.transAxes, fontsize=12)
            ax7.set_title('Spectral Rolloff', fontsize=12, fontweight='bold')
        
        # 保存图片
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"audio_features_{Path(audio_path).stem}_{timestamp}.png"
        filepath = self.output_dir / filename
        
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
        
        return str(filepath)
    
    def create_recognition_result_plot(self, result: RecognitionResult, 
                                     audio_info: AudioInfo, 
                                     engine_stats: Dict[str, Any]) -> str:
        """
        创建识别结果图
        
        Args:
            result: 识别结果
            audio_info: 音频信息
            engine_stats: 引擎统计信息
            
        Returns:
            str: 保存的图片路径
        """
        # 创建图形
        fig = plt.figure(figsize=(14, 10))
        fig.suptitle('Vehicle Recognition Results', fontsize=20, fontweight='bold', color=self.colors['text'])
        
        # 创建网格布局
        gs = GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.3)
        
        # 1. 主要识别结果 (占据第一行)
        ax1 = fig.add_subplot(gs[0, :])
        ax1.axis('off')
        
        # 创建结果卡片
        card_rect = patches.FancyBboxPatch((0.1, 0.2), 0.8, 0.6, 
                                          boxstyle="round,pad=0.05",
                                          facecolor=self.colors['primary'], 
                                          edgecolor='none', alpha=0.1)
        ax1.add_patch(card_rect)
        
        # 主要结果文本
        result_text = f"Identified Vehicle: {result.vehicle_id}\nConfidence: {result.confidence:.1%}"
        ax1.text(0.5, 0.5, result_text, ha='center', va='center', 
                fontsize=24, fontweight='bold', color=self.colors['primary'])
        
        # 处理时间
        time_text = f"Processing Time: {result.processing_time:.3f} seconds"
        ax1.text(0.5, 0.15, time_text, ha='center', va='center', 
                fontsize=14, color=self.colors['text'])
        
        # 2. 置信度仪表盘
        ax2 = fig.add_subplot(gs[1, 0])
        self._create_confidence_gauge(ax2, result.confidence)
        
        # 3. 匹配结果详情
        ax3 = fig.add_subplot(gs[1, 1])
        if hasattr(result, 'matches') and result.matches:
            self._create_matches_chart(ax3, result.matches)
        else:
            ax3.text(0.5, 0.5, 'Detailed Match Info\nNot Available', ha='center', va='center', 
                    transform=ax3.transAxes, fontsize=14, color=self.colors['text'])
            ax3.set_title('Match Details', fontsize=14, fontweight='bold')
        
        # 4. 系统信息
        ax4 = fig.add_subplot(gs[2, 0])
        self._create_system_info_chart(ax4, engine_stats, audio_info)
        
        # 5. 识别历史/统计
        ax5 = fig.add_subplot(gs[2, 1])
        self._create_stats_chart(ax5, engine_stats)
        
        # 保存图片
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"recognition_result_{result.vehicle_id}_{timestamp}.png"
        filepath = self.output_dir / filename
        
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
        
        return str(filepath)
    
    def _create_confidence_gauge(self, ax, confidence: float):
        """创建置信度仪表盘"""
        # 创建半圆仪表盘
        theta = np.linspace(0, np.pi, 100)
        r = 1
        
        # 背景弧
        ax.plot(r * np.cos(theta), r * np.sin(theta), 
               color=self.colors['background'], linewidth=20, alpha=0.3)
        
        # 置信度弧
        confidence_theta = np.linspace(0, confidence * np.pi, int(confidence * 100))
        color = self.colors['success'] if confidence > 0.8 else \
                self.colors['accent'] if confidence > 0.6 else self.colors['secondary']
        
        ax.plot(r * np.cos(confidence_theta), r * np.sin(confidence_theta), 
               color=color, linewidth=20)
        
        # 置信度文本
        ax.text(0, -0.3, f'{confidence:.1%}', ha='center', va='center', 
               fontsize=20, fontweight='bold', color=color)
        ax.text(0, -0.5, 'Confidence', ha='center', va='center', 
               fontsize=12, color=self.colors['text'])
        
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-0.6, 1.2)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title('Recognition Confidence', fontsize=14, fontweight='bold')
    
    def _create_matches_chart(self, ax, matches: List[Dict]):
        """创建匹配结果图表"""
        if len(matches) > 5:
            matches = matches[:5]  # 只显示前5个
        
        vehicles = [m.get('vehicle_id', f'Vehicle_{i}') for i, m in enumerate(matches)]
        scores = [m.get('score', m.get('confidence', 0)) for m in matches]
        
        bars = ax.barh(vehicles, scores, color=self.colors['primary'], alpha=0.7)
        
        # 添加数值标签
        for i, (bar, score) in enumerate(zip(bars, scores)):
            ax.text(score + 0.01, i, f'{score:.3f}', 
                   va='center', fontsize=10, color=self.colors['text'])
        
        ax.set_xlabel('Match Score')
        ax.set_title('Top Match Results', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        ax.set_xlim(0, 1)
    
    def _create_system_info_chart(self, ax, engine_stats: Dict, audio_info: AudioInfo):
        """创建系统信息图表"""
        ax.axis('off')
        
        info_items = [
            ('Audio File', audio_info.file_name),
            ('File Size', f'{audio_info.file_size/1024:.1f} KB'),
        ]
        
        if audio_info.duration:
            info_items.append(('Duration', f'{audio_info.duration:.2f}s'))
        
        if engine_stats:
            if 'model_loaded' in engine_stats:
                info_items.append(('Model Status', 'Loaded' if engine_stats['model_loaded'] else 'Not Loaded'))
            if 'total_vehicles' in engine_stats:
                info_items.append(('Vehicle Database', f"{engine_stats['total_vehicles']} vehicles"))
        
        # 绘制信息表格
        y_pos = 0.9
        for label, value in info_items:
            ax.text(0.05, y_pos, f'{label}:', fontweight='bold', 
                   transform=ax.transAxes, fontsize=11, color=self.colors['text'])
            ax.text(0.55, y_pos, str(value), 
                   transform=ax.transAxes, fontsize=11, color=self.colors['primary'])
            y_pos -= 0.15
        
        ax.set_title('System Information', fontsize=14, fontweight='bold')
    
    def _create_stats_chart(self, ax, engine_stats: Dict):
        """创建统计图表"""
        if not engine_stats or 'recognition_count' not in engine_stats:
            ax.text(0.5, 0.5, 'Statistics\nNot Available', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=14, color=self.colors['text'])
            ax.set_title('Recognition Statistics', fontsize=14, fontweight='bold')
            return
        
        # 创建简单的统计饼图
        labels = ['Successful', 'Failed']
        success_count = engine_stats.get('recognition_count', 1)
        fail_count = engine_stats.get('failed_count', 0)
        sizes = [success_count, fail_count] if fail_count > 0 else [success_count]
        colors = [self.colors['success'], self.colors['secondary']] if fail_count > 0 else [self.colors['success']]
        
        if fail_count > 0:
            ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        else:
            ax.pie([1], labels=['Successful'], colors=[self.colors['success']], autopct='100%', startangle=90)
        
        ax.set_title('Recognition Statistics', fontsize=14, fontweight='bold')
    
    def process_audio(self, audio_path: str) -> Dict[str, str]:
        """
        处理音频文件，生成所有可视化结果
        
        Args:
            audio_path: 音频文件路径
            
        Returns:
            Dict[str, str]: 生成的文件路径
        """
        logger.info(f"开始处理音频: {audio_path}")
        
        # 执行识别
        result, audio_info = self.recognition_engine.recognize(audio_path)
        engine_stats = self.recognition_engine.get_engine_stats()
        
        # 生成可视化
        features_plot = self.create_audio_features_plot(audio_path, audio_info)
        result_plot = self.create_recognition_result_plot(result, audio_info, engine_stats)
        
        # 保存JSON结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_filename = f"recognition_data_{Path(audio_path).stem}_{timestamp}.json"
        json_filepath = self.output_dir / json_filename
        
        result_data = {
            'timestamp': timestamp,
            'audio_info': {
                'file_path': audio_info.file_path,
                'file_name': audio_info.file_name,
                'file_size': audio_info.file_size,
                'duration': audio_info.duration,
                'sample_rate': audio_info.sample_rate
            },
            'recognition_result': {
                'vehicle_id': result.vehicle_id,
                'confidence': result.confidence,
                'processing_time': result.processing_time
            },
            'engine_stats': engine_stats,
            'generated_files': {
                'features_plot': features_plot,
                'result_plot': result_plot,
                'json_data': str(json_filepath)
            }
        }
        
        with open(json_filepath, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"处理完成，结果保存至: {self.output_dir}")
        
        return {
            'features_plot': features_plot,
            'result_plot': result_plot,
            'json_data': str(json_filepath)
        }


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='增强版车辆声纹识别可视化')
    parser.add_argument('audio_path', help='音频文件路径')
    parser.add_argument('--output', '-o', default='results', help='输出目录')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.audio_path):
        print(f"音频文件不存在: {args.audio_path}")
        return
    
    try:
        visualizer = EnhancedVisualizer(args.output)
        results = visualizer.process_audio(args.audio_path)
        
        print("可视化完成")
        print(f"特征图: {results['features_plot']}")
        print(f"结果图: {results['result_plot']}")
        print(f"数据: {results['json_data']}")
        
    except Exception as e:
        logger.error(f"处理失败: {e}")
        print(f"错误: {e}")


if __name__ == '__main__':
    main()