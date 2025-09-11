#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
三音频混合特征提取工具
使用vector_similarity_engine.py的方法提取特征，只保留特征混合功能
"""

import os
import random
import numpy as np
import json
import sys
sys.path.append('.')  # 添加当前目录到Python路径
from vector_similarity_engine import VectorSimilarityEngine
from sklearn.metrics.pairwise import cosine_similarity

def mix_three_feature_vectors(features, mix_ratios=None):
    """混合三个特征向量"""
    if len(features) != 3:
        raise ValueError("需要恰好三个特征向量进行混合")
    
    # 默认等比例混合
    if mix_ratios is None:
        mix_ratios = [1/3, 1/3, 1/3]
    
    # 统一维度（取最短的长度）
    min_length = min(len(f) for f in features)
    normalized_features = [f[:min_length] for f in features]
    
    # 线性混合
    mixed_feature = np.zeros(min_length)
    for feat, ratio in zip(normalized_features, mix_ratios):
        mixed_feature += ratio * feat
    
    # 归一化
    mixed_feature = mixed_feature / np.linalg.norm(mixed_feature)
    
    return mixed_feature

def extract_features_from_audio(audio_path):
    """使用vector_similarity_engine的方法提取音频特征"""
    engine = VectorSimilarityEngine()
    
    # 提取特征
    features_dict = engine.feature_extractor.extract_features(audio_path)
    
    if not features_dict or 'fused_features' not in features_dict:
        raise ValueError(f"无法从音频文件 {audio_path} 提取特征")
    
    features = features_dict['fused_features']
    if features is None or len(features) == 0:
        raise ValueError(f"从音频文件 {audio_path} 提取的特征为空")
    
    return features

def main():
    """主函数"""
    print("🎵 三音频混合特征提取工具")
    print("=" * 50)
    
    # 加载数据库
    try:
        with open('models/similarity_database.json', 'r') as f:
            data = json.load(f)
        vehicle_profiles = data['vehicle_profiles']
    except Exception as e:
        print(f"❌ 加载数据库失败: {e}")
        return
    
    print(f"📊 数据库中有 {len(vehicle_profiles)} 辆车的特征")
    
    # 随机选择三个车辆
    vehicle_ids = list(vehicle_profiles.keys())
    if len(vehicle_ids) < 3:
        print("❌ 数据库中的车辆数量不足，需要至少3辆")
        return
    
    selected_vehicles = random.sample(vehicle_ids, 3)
    
    print(f"\n🔀 随机选择的车辆:")
    for i, vehicle_id in enumerate(selected_vehicles, 1):
        vehicle_type = vehicle_profiles[vehicle_id]['metadata']['vehicle_type']
        print(f"  车辆{i}: {vehicle_id} ({vehicle_type})")
    
    # 获取特征向量
    features = []
    for vehicle_id in selected_vehicles:
        feature = np.array(vehicle_profiles[vehicle_id]['features'])
        features.append(feature)
        print(f"  {vehicle_id}: {len(feature)}维")
    
    # 混合特征向量（等比例混合）
    print(f"\n🔀 混合三个特征向量...")
    mixed_feature = mix_three_feature_vectors(features)
    print(f"  混合特征维度: {len(mixed_feature)}维")
    print(f"  混合特征范数: {np.linalg.norm(mixed_feature):.6f}")
    
    # 计算与所有车辆的相似度
    print(f"\n🎯 计算相似度...")
    similarities = {}
    for vehicle_id, profile in vehicle_profiles.items():
        db_feature = np.array(profile['features'])
        
        # 确保维度匹配
        min_dim = min(len(mixed_feature), len(db_feature))
        mixed_trimmed = mixed_feature[:min_dim]
        db_trimmed = db_feature[:min_dim]
        
        similarity = cosine_similarity(
            mixed_trimmed.reshape(1, -1),
            db_trimmed.reshape(1, -1)
        )[0][0]
        
        similarities[vehicle_id] = similarity
    
    # 按相似度排序
    sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    
    # 显示识别结果
    print(f"\n🏆 相似度排名 (前10):")
    print("-" * 60)
    print(f"{'排名':<4} {'车辆ID':<15} {'车辆类型':<10} {'相似度':<8}")
    print("-" * 60)
    
    for rank, (vehicle_id, score) in enumerate(sorted_similarities[:10], 1):
        vehicle_type = vehicle_profiles[vehicle_id]['metadata']['vehicle_type']
        is_original = "✅" if vehicle_id in selected_vehicles else ""
        print(f"{rank:<4} {vehicle_id:<15} {vehicle_type:<10} {score:.4f} {is_original}")
    
    print("-" * 60)
    
    # 显示原始车辆的排名
    print(f"\n📊 原始车辆识别结果:")
    for vehicle_id in selected_vehicles:
        for rank, (vid, score) in enumerate(sorted_similarities, 1):
            if vid == vehicle_id:
                vehicle_type = vehicle_profiles[vehicle_id]['metadata']['vehicle_type']
                print(f"  {vehicle_id} ({vehicle_type}): 排名第{rank}位, 相似度: {score:.4f}")
                break
    
    # 计算识别准确率
    success_count = sum(1 for vehicle_id in selected_vehicles 
                       for rank, (vid, _) in enumerate(sorted_similarities, 1)
                       if vid == vehicle_id and rank <= 5)
    accuracy = success_count / len(selected_vehicles)
    
    print(f"\n📈 识别准确率 (前5名内): {accuracy:.2%} ({success_count}/{len(selected_vehicles)})")
    
    print(f"\n✅ 特征混合和识别完成！")

if __name__ == "__main__":
    main()
