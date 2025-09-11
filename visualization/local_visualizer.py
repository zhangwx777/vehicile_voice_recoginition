#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
本地车辆声纹识别可视化工具 (兼容版本)
使用新的模块化架构
"""

import os
import sys
import argparse
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from visualization.enhanced_visualizer import EnhancedVisualizer
from core.logger import logger

class LocalVehicleVisualizer:
    """
    本地车辆声纹识别可视化器 (兼容版本)
    使用新的增强可视化器
    """
    
    def __init__(self, output_dir="results/visualizations"):
        """
        初始化可视化器
        
        Args:
            output_dir: 输出目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 使用新的增强可视化器
        self.visualizer = EnhancedVisualizer(output_dir=output_dir)
    
    def visualize_audio_features(self, audio_path, save_prefix=None):
        """
        可视化音频特征 (兼容方法)
        
        Args:
            audio_path: 音频文件路径
            save_prefix: 保存文件前缀
            
        Returns:
            dict: 包含保存的图片路径
        """
        # 使用新的增强可视化器处理
        results = self.visualizer.process_audio(audio_path)
        return {
            'features_plot': results['features_plot'],
            'result_plot': results['result_plot']
        }
    
    def visualize_recognition_result(self, result, audio_path, save_prefix=None):
        """
        可视化识别结果 (兼容方法)
        
        Args:
            result: 识别结果
            audio_path: 音频文件路径
            save_prefix: 保存文件前缀
            
        Returns:
            str: 保存的图片路径
        """
        # 这个方法现在集成在process_audio中
        results = self.visualizer.process_audio(audio_path)
        return results['result_plot']
    
    def recognize_and_visualize(self, audio_path, mode='similarity'):
        """
        识别音频并生成可视化结果 (兼容方法)
        
        Args:
            audio_path: 音频文件路径
            mode: 识别模式
            
        Returns:
            dict: 包含识别结果和保存的图片路径
        """
        if not os.path.exists(audio_path):
            logger.error(f"File not found: {audio_path}")
            return None
        
        try:
            # 使用新的增强可视化器
            results = self.visualizer.process_audio(audio_path)
            
            logger.info(f"Recognition and visualization completed")
            logger.info(f"Audio features plot: {results['features_plot']}")
            logger.info(f"Recognition result plot: {results['result_plot']}")
            logger.info(f"Data file: {results['json_data']}")
            
            return {
                'visualizations': results,
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Processing failed: {e}")
            return None

def main():
    """
    主函数 - 命令行接口
    """
    parser = argparse.ArgumentParser(description='Local Vehicle Voice Recognition Visualizer')
    parser.add_argument('audio_path', help='Path to audio file')
    parser.add_argument('--mode', choices=['similarity'], 
                       default='similarity', help='识别模式（仅支持车体相似度识别）')
    parser.add_argument('--output', '-o', default='results/visualizations', 
                       help='Output directory for visualizations')
    
    args = parser.parse_args()
    
    try:
        # 创建可视化器
        visualizer = LocalVehicleVisualizer(output_dir=args.output)
        
        # 执行识别和可视化
        result = visualizer.recognize_and_visualize(args.audio_path, mode=args.mode)
        
        if not result:
            logger.error("Processing failed!")
            sys.exit(1)
        else:
            logger.info("Processing completed!")
            
    except Exception as e:
        logger.error(f"Program execution failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()