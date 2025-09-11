#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试数据库功能脚本
直接使用现有的数据库进行测试
"""

import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def test_database():
    """测试数据库功能"""
    print("🔍 测试车辆声纹识别数据库功能")
    print("-" * 40)
    
    # 加载数据库
    try:
        with open('models/similarity_database.json', 'r') as f:
            data = json.load(f)
        
        vehicle_profiles = data['vehicle_profiles']
        print(f"数据库中的车辆数量: {len(vehicle_profiles)}")
        
        # 显示车辆类型统计
        vehicle_types = {}
        for vehicle_id, profile in vehicle_profiles.items():
            vehicle_type = profile['metadata']['vehicle_type']
            vehicle_types[vehicle_type] = vehicle_types.get(vehicle_type, 0) + 1
        
        print("\n📊 车辆类型统计:")
        for v_type, count in vehicle_types.items():
            print(f"  {v_type}: {count}辆")
        
        # 测试特征向量相似度计算
        print("\n🧪 特征向量相似度测试:")
        
        # 获取两个bus的特征向量
        bus1_features = np.array(vehicle_profiles['bus_000']['features']).reshape(1, -1)
        bus2_features = np.array(vehicle_profiles['bus_001']['features']).reshape(1, -1)
        
        # 计算相似度
        similarity = cosine_similarity(bus1_features, bus2_features)[0][0]
        print(f"bus_000 和 bus_001 的相似度: {similarity:.4f}")
        
        # 测试不同类型车辆的相似度
        bus_features = np.array(vehicle_profiles['bus_000']['features']).reshape(1, -1)
        motorcycle_features = np.array(vehicle_profiles['motorcycle_000']['features']).reshape(1, -1)
        
        cross_similarity = cosine_similarity(bus_features, motorcycle_features)[0][0]
        print(f"bus_000 和 motorcycle_000 的相似度: {cross_similarity:.4f}")
        
        print("\n✅ 数据库功能测试成功!")
        print(f"特征向量维度: {len(bus1_features[0])}")
        print(f"相似度阈值配置: {data['config']['similarity_threshold']}")
        
    except Exception as e:
        print(f"❌ 数据库测试失败: {e}")

if __name__ == "__main__":
    test_database()
