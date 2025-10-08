#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
本地车辆声纹识别系统
提供实时音频识别和数据库管理功能
"""

# 使用公共工具模块统一导入
from core.common_utils import (
    os, sys, time, json, Path,
    List, Dict, Optional, Tuple, Any, Union,
    dataclass, asdict, warnings, np,
    config_manager, file_manager, logger_factory, TimerContext
)

import argparse

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.unified_engine import UnifiedVehicleRecognitionEngine
from core.logger import log_system_startup, log_system_shutdown
from audio_visualizer import AudioVisualizer
from core.exceptions import handle_exceptions, safe_execute, log_system_error

def main():
    """主函数"""
    # 记录系统启动
    log_system_startup()
    
    parser = argparse.ArgumentParser(description='车辆声纹识别系统')
    parser.add_argument('audio_file', help='音频文件路径')
    parser.add_argument('--mode', choices=['ecapa', 'mfcc'], default='ecapa', 
                       help='识别模式 (默认: ecapa)')
    parser.add_argument('--visualize', action='store_true', default=True,
                       help='生成可视化图表（默认启用）')
    parser.add_argument('--no-visualize', action='store_true',
                       help='禁用可视化图表')
    
    args = parser.parse_args()
    
    # 处理可视化参数
    if args.no_visualize:
        args.visualize = False
    
    # 检查文件是否存在
    if not os.path.exists(args.audio_file):
        print(f"错误: 文件不存在 {args.audio_file}")
        sys.exit(1)
    
    print("正在初始化车辆识别引擎...")
    
    try:
        # 初始化识别引擎
        engine = UnifiedVehicleRecognitionEngine()
        
        # 执行识别
        print(f"识别文件: {args.audio_file}")
        
        # 普通识别模式
        result, audio_info = engine.recognize(args.audio_file)
        
        # 显示识别结果
        if result.vehicle_id != "unknown":
            print(f"识别成功: 车辆 {result.vehicle_id} (置信度: {result.confidence:.4f})")
            if hasattr(result, 'similar_vehicles') and result.similar_vehicles:
                print("相似度排名:")
                for i, (vid, score) in enumerate(result.similar_vehicles[:5], 1):
                    print(f"  {i}. 车辆{vid}: {score:.4f}")
        else:
            print("识别失败: 未找到匹配车辆")
        
        # 可视化处理
        if args.visualize:
            print("\n生成可视化结果...")
            try:
                visualizer = AudioVisualizer()
                
                # 准备数据
                result_data = {
                    'vehicle_id': result.vehicle_id,
                    'confidence': result.confidence,
                    'method': result.method,
                    'processing_time': getattr(result, 'processing_time', 0),
                    'similar_vehicles': getattr(result, 'similar_vehicles', [])
                }
                
                audio_file_path = Path(args.audio_file)
                audio_info_data = {
                    'file_name': audio_file_path.name,
                    'file_size': audio_file_path.stat().st_size if audio_file_path.exists() else 0,
                    'duration': getattr(audio_info, 'duration', 0),
                    'sample_rate': getattr(audio_info, 'sample_rate', 16000),
                    'channels': getattr(audio_info, 'channels', 1)
                }
                
                # 生成可视化图表
                features_path = visualizer.visualize_audio_features(args.audio_file)
                results_path = visualizer.visualize_recognition_result(result_data, audio_info_data)
                
                print(f"音频特征: {features_path}")
                print(f"识别结果: {results_path}")
                
            except Exception as e:
                log_system_error(e, "音频可视化处理", include_traceback=False)
                print(f"可视化失败: {e}")
                
    except Exception as e:
        log_system_error(e, "程序主流程执行", include_traceback=True)
        print(f"程序执行失败: {e}")
        sys.exit(1)
    finally:
        # 记录系统关闭
        log_system_shutdown()

if __name__ == "__main__":
    main()