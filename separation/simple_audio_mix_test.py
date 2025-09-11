#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化版音频混合测试
直接使用数据库中的特征向量进行混合和识别测试
"""

import os
import random
import numpy as np
import json
from sklearn.metrics.pairwise import cosine_similarity

def load_audio_database():
    """加载音频数据库信息"""
    try:
        with open('models/similarity_database.json', 'r') as f:
            data = json.load(f)
        return data['vehicle_profiles']
    except Exception as e:
        print(f"❌ 加载数据库失败: {e}")
        return {}

def mix_feature_vectors(feature1, feature2, mix_ratio=0.5):
    """混合两个特征向量"""
    if len(feature1) != len(feature2):
        # 如果维度不同，取较短的长度
        min_length = min(len(feature1), len(feature2))
        feature1 = feature1[:min_length]
        feature2 = feature2[:min_length]
    
    # 混合特征向量
    mixed_feature = mix_ratio * feature1 + (1 - mix_ratio) * feature2
    
    # 归一化
    mixed_feature = mixed_feature / np.linalg.norm(mixed_feature)
    
    return mixed_feature

def main():
    """主函数"""
    print("🎵 简化版音频混合识别测试")
    print("=" * 50)
    
    # 加载数据库
    vehicle_profiles = load_audio_database()
    if not vehicle_profiles:
        return
    
    print(f"📊 数据库中有 {len(vehicle_profiles)} 辆车的特征")
    
    # 随机选择两个车辆
    vehicle_ids = list(vehicle_profiles.keys())
    if len(vehicle_ids) < 2:
        print("❌ 数据库中的车辆数量不足")
        return
    
    selected_vehicles = random.sample(vehicle_ids, 2)
    
    print(f"\n🔀 随机选择的车辆:")
    for i, vehicle_id in enumerate(selected_vehicles, 1):
        vehicle_type = vehicle_profiles[vehicle_id]['metadata']['vehicle_type']
        print(f"  车辆{i}: {vehicle_id} ({vehicle_type})")
    
    # 获取特征向量
    feature1 = np.array(vehicle_profiles[selected_vehicles[0]]['features'])
    feature2 = np.array(vehicle_profiles[selected_vehicles[1]]['features'])
    
    print(f"\n📏 特征向量维度:")
    print(f"  {selected_vehicles[0]}: {len(feature1)}维")
    print(f"  {selected_vehicles[1]}: {len(feature2)}维")
    
    # 混合特征向量
    print(f"\n🔀 混合特征向量...")
    mixed_feature = mix_feature_vectors(feature1, feature2, mix_ratio=0.5)
    print(f"  混合特征维度: {len(mixed_feature)}维")
    
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
    
    # 显示原始车辆之间的相似度
    print(f"\n🔍 原始车辆间相似度:")
    original_similarity = cosine_similarity(
        feature1.reshape(1, -1),
        feature2.reshape(1, -1)
    )[0][0]
    print(f"  {selected_vehicles[0]} 和 {selected_vehicles[1]} 的原始相似度: {original_similarity:.4f}")

if __name__ == "__main__":
    main()
