#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
音频混合识别测试脚本
随机抽取两个音频混合，再通过提取的混合音频的向量特征对比识别出原本的音频
"""

import os
import random
import numpy as np
import librosa
import json
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from enhanced_feature_extractor import EnhancedAudioFeatureExtractor

def load_audio_database():
    """加载音频数据库信息"""
    try:
        with open('../models/similarity_database.json', 'r') as f:
            data = json.load(f)
        return data['vehicle_profiles']
    except Exception as e:
        print(f"❌ 加载数据库失败: {e}")
        return {}

def get_random_audio_files(vehicle_profiles, count=2):
    """随机获取音频文件"""
    audio_files = []
    vehicle_ids = []
    
    # 获取所有可用的车辆ID
    available_vehicles = list(vehicle_profiles.keys())
    if len(available_vehicles) < count:
        print(f"❌ 数据库中的车辆数量不足，需要至少{count}辆")
        return [], []
    
    # 随机选择车辆
    selected_vehicles = random.sample(available_vehicles, count)
    
    for vehicle_id in selected_vehicles:
        # 构建音频文件路径
        vehicle_type = vehicle_profiles[vehicle_id]['metadata']['vehicle_type']
        audio_dir = f"vehicle_audio_data/individual_recognition/individual_vehicles/{vehicle_id}"
        
        if os.path.exists(audio_dir):
            # 获取该车辆的所有音频文件
            audio_files_list = [f for f in os.listdir(audio_dir) if f.endswith('.wav')]
            if audio_files_list:
                # 随机选择一个音频文件
                selected_audio = random.choice(audio_files_list)
                audio_path = os.path.join(audio_dir, selected_audio)
                audio_files.append(audio_path)
                vehicle_ids.append(vehicle_id)
            else:
                print(f"⚠️ 车辆 {vehicle_id} 没有音频文件")
        else:
            print(f"⚠️ 车辆目录不存在: {audio_dir}")
    
    return audio_files, vehicle_ids

def mix_audio_files(audio_files, mix_ratio=0.5):
    """混合两个音频文件"""
    if len(audio_files) != 2:
        raise ValueError("需要恰好两个音频文件进行混合")
    
    # 加载第一个音频
    audio1, sr1 = librosa.load(audio_files[0], sr=None)
    # 加载第二个音频
    audio2, sr2 = librosa.load(audio_files[1], sr=None)
    
    # 统一采样率
    target_sr = max(sr1, sr2)
    if sr1 != target_sr:
        audio1 = librosa.resample(audio1, orig_sr=sr1, target_sr=target_sr)
    if sr2 != target_sr:
        audio2 = librosa.resample(audio2, orig_sr=sr2, target_sr=target_sr)
    
    # 统一长度（取较短的长度）
    min_length = min(len(audio1), len(audio2))
    audio1 = audio1[:min_length]
    audio2 = audio2[:min_length]
    
    # 混合音频
    mixed_audio = mix_ratio * audio1 + (1 - mix_ratio) * audio2
    
    # 归一化音量
    if np.max(np.abs(mixed_audio)) > 0:
        mixed_audio = mixed_audio / np.max(np.abs(mixed_audio)) * 0.8
    
    return mixed_audio, target_sr, audio_files

def extract_features_from_mixed(audio_data, sample_rate):
    """从混合音频中提取特征"""
    extractor = EnhancedAudioFeatureExtractor(
        target_sr=16000,
        feature_dim=512,
        use_pca=True,
        pca_components=256,
        normalize_features=True
    )
    
    # 提取特征
    result = extractor.extract_features((audio_data, sample_rate))
    return result

def recognize_original_vehicles(mixed_features, vehicle_profiles, original_vehicle_ids):
    """识别原始车辆"""
    if mixed_features is None or mixed_features.get('fused_features') is None:
        print("❌ 无法提取混合音频特征")
        return None, None
    
    mixed_feature_vector = mixed_features['fused_features']
    
    # 计算与所有车辆的相似度
    similarities = {}
    for vehicle_id, profile in vehicle_profiles.items():
        db_feature_vector = np.array(profile['features'])
        
        # 计算余弦相似度
        similarity = cosine_similarity(
            mixed_feature_vector.reshape(1, -1),
            db_feature_vector.reshape(1, -1)
        )[0][0]
        
        similarities[vehicle_id] = similarity
    
    # 按相似度排序
    sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    
    # 检查原始车辆是否在相似度排名中
    original_ranks = {}
    for original_id in original_vehicle_ids:
        for rank, (vehicle_id, score) in enumerate(sorted_similarities, 1):
            if vehicle_id == original_id:
                original_ranks[original_id] = (rank, score)
                break
    
    return sorted_similarities, original_ranks

def main():
    """主函数"""
    print("🎵 音频混合识别测试")
    print("=" * 50)
    
    # 加载数据库
    vehicle_profiles = load_audio_database()
    if not vehicle_profiles:
        return
    
    print(f"📊 数据库中有 {len(vehicle_profiles)} 辆车的特征")
    
    # 随机选择两个音频文件
    audio_files, vehicle_ids = get_random_audio_files(vehicle_profiles, 2)
    if len(audio_files) != 2:
        return
    
    print(f"\n🔀 随机选择的车辆:")
    for i, (audio_file, vehicle_id) in enumerate(zip(audio_files, vehicle_ids), 1):
        vehicle_type = vehicle_profiles[vehicle_id]['metadata']['vehicle_type']
        print(f"  车辆{i}: {vehicle_id} ({vehicle_type})")
        print(f"    音频文件: {os.path.basename(audio_file)}")
    
    # 混合音频
    print(f"\n🔊 混合音频...")
    mixed_audio, sample_rate, _ = mix_audio_files(audio_files, mix_ratio=0.5)
    print(f"  混合音频长度: {len(mixed_audio)/sample_rate:.2f}秒")
    print(f"  采样率: {sample_rate}Hz")
    
    # 从混合音频中提取特征
    print(f"\n🔍 提取混合音频特征...")
    mixed_features = extract_features_from_mixed(mixed_audio, sample_rate)
    
    if mixed_features is None:
        print("❌ 特征提取失败")
        return
    
    print(f"  融合特征维度: {mixed_features['feature_dim']}")
    
    # 识别原始车辆
    print(f"\n🎯 识别原始车辆...")
    similarities, original_ranks = recognize_original_vehicles(
        mixed_features, vehicle_profiles, vehicle_ids
    )
    
    if similarities is None:
        return
    
    # 显示识别结果
    print(f"\n🏆 相似度排名 (前10):")
    print("-" * 60)
    print(f"{'排名':<4} {'车辆ID':<15} {'车辆类型':<10} {'相似度':<8}")
    print("-" * 60)
    
    for rank, (vehicle_id, score) in enumerate(similarities[:10], 1):
        vehicle_type = vehicle_profiles[vehicle_id]['metadata']['vehicle_type']
        is_original = "✅" if vehicle_id in vehicle_ids else ""
        print(f"{rank:<4} {vehicle_id:<15} {vehicle_type:<10} {score:.4f} {is_original}")
    
    print("-" * 60)
    
    # 显示原始车辆的排名
    print(f"\n📊 原始车辆识别结果:")
    for vehicle_id in vehicle_ids:
        if vehicle_id in original_ranks:
            rank, score = original_ranks[vehicle_id]
            vehicle_type = vehicle_profiles[vehicle_id]['metadata']['vehicle_type']
            print(f"  {vehicle_id} ({vehicle_type}): 排名第{rank}位, 相似度: {score:.4f}")
        else:
            print(f"  {vehicle_id}: 未找到")
    
    # 计算识别准确率
    success_count = sum(1 for vehicle_id in vehicle_ids 
                       if vehicle_id in original_ranks and original_ranks[vehicle_id][0] <= 5)
    accuracy = success_count / len(vehicle_ids)
    
    print(f"\n📈 识别准确率 (前5名内): {accuracy:.2%} ({success_count}/{len(vehicle_ids)})")

if __name__ == "__main__":
    main()
