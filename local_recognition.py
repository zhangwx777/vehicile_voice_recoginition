#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
本地车辆声纹识别脚本
提供简单的命令行接口进行音频识别
"""

import os
import sys
import argparse
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.unified_engine import UnifiedVehicleRecognitionEngine, RecognitionMode
from config import SYSTEM_MODE_CONFIG
from visualization.enhanced_visualizer import EnhancedVisualizer

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='车辆声纹识别本地版')
    parser.add_argument('audio_file', help='音频文件路径')
    parser.add_argument('--mode', choices=['similarity'], 
                       default='similarity', help='识别模式（仅支持车体相似度识别）')
    parser.add_argument('--threshold', type=float, default=0.7, help='置信度阈值')
    parser.add_argument('--no-viz', action='store_true', help='禁用可视化功能')
    parser.add_argument('--viz-only', action='store_true', help='仅生成可视化，不显示文本结果')
    
    args = parser.parse_args()
    
    # 检查音频文件是否存在
    if not os.path.exists(args.audio_file):
        print(f"错误: 音频文件 {args.audio_file} 不存在")
        return
    
    try:
        # 初始化识别引擎
        engine = UnifiedVehicleRecognitionEngine()
        
        # 设置识别模式（仅支持相似度搜索）
        mode_map = {
            'similarity': RecognitionMode.SIMILARITY
        }
        
        # 执行识别
        print(f"识别文件: {os.path.basename(args.audio_file)} (模式: {args.mode})")
        print("-" * 40)
        
        result, audio_info = engine.recognize(
            audio_path=args.audio_file,
            confidence_threshold=args.threshold
        )
        
        # 获取分离的向量结果
        if hasattr(engine, 'similarity_engine') and engine.similarity_engine:
            try:
                # 提取查询音频的分离特征
                query_features = engine.similarity_engine.feature_extractor.extract_features(args.audio_file)
                if query_features:
                    print("\n🔍 分离向量分析:")
                    print("-" * 30)
                    
                    # 显示YAMNet特征信息
                    if query_features.get('yamnet_features') is not None:
                        yamnet_feat = query_features['yamnet_features']
                        print(f"YAMNet特征维度: {len(yamnet_feat)}")
                        print(f"YAMNet特征范围: [{yamnet_feat.min():.4f}, {yamnet_feat.max():.4f}]")
                        print(f"YAMNet特征均值: {yamnet_feat.mean():.4f}")
                    
                    # 显示VGGish特征信息
                    if query_features.get('vggish_features') is not None:
                        vggish_feat = query_features['vggish_features']
                        print(f"VGGish特征维度: {len(vggish_feat)}")
                        print(f"VGGish特征范围: [{vggish_feat.min():.4f}, {vggish_feat.max():.4f}]")
                        print(f"VGGish特征均值: {vggish_feat.mean():.4f}")
                    
                    # 显示融合特征信息
                    if query_features.get('fused_features') is not None:
                        fused_feat = query_features['fused_features']
                        print(f"融合特征维度: {len(fused_feat)}")
                        print(f"融合特征范围: [{fused_feat.min():.4f}, {fused_feat.max():.4f}]")
                        print(f"融合特征均值: {fused_feat.mean():.4f}")
                    
                    print("-" * 30)
                    
                    # 与数据库中车辆特征的对比
                    if result.vehicle_id and result.vehicle_id != "unknown":
                        try:
                            # 获取数据库中匹配车辆的特征
                            db_vehicle_info = engine.similarity_engine.get_vehicle_info(result.vehicle_id)
                            if db_vehicle_info:
                                print("\n📊 特征对比分析:")
                                print("-" * 30)
                                
                                # 获取数据库中该车辆的特征向量
                                db_profile = engine.similarity_engine.vehicle_profiles[result.vehicle_id]
                                db_feature_vector = db_profile.features
                                
                                if query_features:
                                    # 计算特征相似度
                                    from sklearn.metrics.pairwise import cosine_similarity
                                    
                                    # 融合特征相似度（直接使用数据库中的融合特征）
                                    if query_features.get('fused_features') is not None:
                                        query_fused = query_features['fused_features']
                                        fused_sim = cosine_similarity(
                                            query_fused.reshape(1, -1),
                                            db_feature_vector.reshape(1, -1)
                                        )[0][0]
                                        print(f"融合特征相似度: {fused_sim:.4f}")
                                        print(f"最终置信度: {result.confidence:.4f}")
                                    
                                    print("-" * 30)
                                    
                                    # 显示特征向量对比摘要
                                    print("\n📈 特征向量对比摘要:")
                                    print("-" * 30)
                                    print(f"查询音频特征维度: {len(query_fused) if 'fused_features' in query_features else 'N/A'}")
                                    print(f"数据库特征维度: {len(db_feature_vector)}")
                                    print(f"特征匹配度: {result.confidence:.4f}")
                                    print("-" * 30)
                                    
                        except Exception as compare_error:
                            print(f"⚠️ 特征对比分析失败: {compare_error}")
                    
            except Exception as e:
                print(f"⚠️ 分离向量分析失败: {e}")
        
        # 生成可视化（默认启用）
        if not args.no_viz:
            try:
                visualizer = EnhancedVisualizer()
                viz_results = visualizer.process_audio(args.audio_file)
                
                if not args.viz_only:
                    print("\n📊 可视化结果已生成:")
                    if 'features_plot' in viz_results:
                        print(f"  音频特征图: {viz_results['features_plot']}")
                    if 'result_plot' in viz_results:
                        print(f"  识别结果图: {viz_results['result_plot']}")
                    if 'json_data' in viz_results:
                        print(f"  详细数据: {viz_results['json_data']}")
                        
            except Exception as viz_error:
                if not args.viz_only:
                    print(f"⚠️ 可视化生成失败: {viz_error}")
        
        # 显示文本结果（除非仅生成可视化）
        if not args.viz_only:
            if result.vehicle_id:
                print("\n✅ 识别成功!")
                print(f"车辆ID: {result.vehicle_id}")
                print(f"置信度: {result.confidence:.4f}")
                print(f"识别方法: {result.method}")
                print(f"处理时间: {result.processing_time:.3f}秒")
                
                if result.similar_vehicles:
                    print("\n相似度排名:")
                    for i, (vid, score) in enumerate(result.similar_vehicles[:5], 1):
                        print(f"  {i}. 车辆{vid}: {score:.4f}")
                        
            else:
                print("❌ 识别失败")
                print(f"置信度过低: {result.confidence:.4f}")
            
    except Exception as e:
        print(f"❌ 系统错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
