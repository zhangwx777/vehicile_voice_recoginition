# scripts/generate_audio_data.py

import os
import numpy as np
import soundfile as sf
from pathlib import Path
from logger import logger
from settings import PATH_CONFIG, AUDIO_CONFIG

def generate_simple_vehicle_sounds(vehicle_type, sample_idx, duration=4, sr=16000):
    """生成简化但特征明显的车辆声音"""
    t = np.linspace(0, duration, int(sr * duration))
    
    # 简化的车辆参数，确保明显区别
    if vehicle_type == 'sedan':
        # 轿车：平稳，中频
        base_freq = 100 + sample_idx % 20  # 100-120Hz
        harmonics = [1, 2, 3]
        weights = [1.0, 0.4, 0.2]
        noise_level = 0.02
        
    elif vehicle_type == 'suv':
        # SUV：厚重，中低频
        base_freq = 80 + sample_idx % 15   # 80-95Hz  
        harmonics = [1, 2, 3, 4]
        weights = [1.0, 0.6, 0.3, 0.15]
        noise_level = 0.025
        
    elif vehicle_type == 'truck':
        # 卡车：深沉，低频
        base_freq = 50 + sample_idx % 20   # 50-70Hz
        harmonics = [1, 2, 3, 4, 6]
        weights = [1.0, 0.7, 0.5, 0.3, 0.2]
        noise_level = 0.04
        
    elif vehicle_type == 'motorcycle':
        # 摩托车：尖锐，高频
        base_freq = 200 + sample_idx % 30  # 200-230Hz
        harmonics = [1, 2, 3, 5, 7]
        weights = [1.0, 0.5, 0.7, 0.4, 0.3]
        noise_level = 0.03
        
    else:  # bus
        # 公交车：低沉，宽频
        base_freq = 60 + sample_idx % 15   # 60-75Hz
        harmonics = [1, 2, 3, 4]
        weights = [1.0, 0.5, 0.3, 0.2]
        noise_level = 0.035
    
    # 生成基础信号
    signal = np.zeros_like(t)
    
    # 简单的转速变化
    rpm_variation = 1 + 0.1 * np.sin(2 * np.pi * 0.3 * t)
    freq_t = base_freq * rpm_variation
    
    # 添加谐波
    for harmonic, weight in zip(harmonics, weights):
        signal += weight * np.sin(2 * np.pi * freq_t * harmonic * t)
    
    # 添加特征性噪声
    if vehicle_type == 'motorcycle':
        # 摩托车加入高频噪声
        signal += 0.2 * np.sin(2 * np.pi * 500 * t) * np.exp(-t)
    elif vehicle_type == 'truck':
        # 卡车加入低频隆隆声
        signal += 0.3 * np.sin(2 * np.pi * 30 * t)
    
    # 添加随机噪声
    signal += noise_level * np.random.randn(len(t))
    
    # 包络
    envelope = 0.5 + 0.5 * np.exp(-((t - duration/2) / (duration/2))**2)
    signal *= envelope
    
    # 归一化
    max_val = np.max(np.abs(signal))
    if max_val > 0:
        signal = signal / max_val * 0.8
    
    return signal

def generate_dataset_fast(num_classes=5, samples_per_class=100, output_dir=None):
    """快速生成数据集"""
    
    output_dir = output_dir or PATH_CONFIG['data_dir']
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    vehicle_types = ['sedan', 'suv', 'truck', 'motorcycle', 'bus'][:num_classes]
    
    logger.info(f"🚀 开始快速生成音频数据集...")
    logger.info(f"📂 输出目录: {output_path}")
    logger.info(f"🚗 车辆类型: {vehicle_types}")
    logger.info(f"📊 每类样本数: {samples_per_class}")
    
    total_generated = 0
    
    for vehicle_type in vehicle_types:
        class_dir = output_path / vehicle_type
        class_dir.mkdir(exist_ok=True)
        
        logger.info(f"🔧 正在生成 {vehicle_type} 类别...")
        
        for sample_idx in range(samples_per_class):
            try:
                # 生成音频
                signal = generate_simple_vehicle_sounds(
                    vehicle_type, sample_idx, 
                    duration=AUDIO_CONFIG['duration'], 
                    sr=AUDIO_CONFIG['sample_rate']
                )
                
                # 保存文件
                filename = class_dir / f"{vehicle_type}_{sample_idx+1:03d}.wav"
                sf.write(filename, signal, AUDIO_CONFIG['sample_rate'])
                
                total_generated += 1
                
                # 每10个打印进度
                if (sample_idx + 1) % 10 == 0:
                    logger.info(f"  ✅ {vehicle_type}: {sample_idx + 1}/{samples_per_class}")
                    
            except Exception as e:
                logger.error(f"❌ 生成失败 {vehicle_type}_{sample_idx}: {str(e)}")
                continue
    
    logger.info(f"🎉 数据集生成完成！总计 {total_generated} 个文件")
    
    # 生成统计信息
    stats_file = output_path / "dataset_stats.txt"
    with open(stats_file, 'w', encoding='utf-8') as f:
        f.write("车载语音识别数据集统计\n")
        f.write("="*40 + "\n")
        f.write(f"总文件数: {total_generated}\n")
        f.write(f"总时长: {total_generated * 4 / 60:.1f} 分钟\n\n")
        f.write("各类别统计:\n")
        for vehicle_type in vehicle_types:
            f.write(f"  {vehicle_type}: {samples_per_class} 个文件\n")
    
    return output_path

def main():
    """主函数"""
    logger.info("🚗 车辆语音识别系统 - 简化数据生成")
    
    try:
        # 快速生成数据集
        output_path = generate_dataset_fast(
            num_classes=5,
            samples_per_class=100
        )
        
        logger.info(f"✅ 数据生成完成！数据保存在: {output_path}")
        logger.info("🎯 下一步: 运行 python train_monitor.py 开始训练")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 数据生成失败: {str(e)}")
        return False

if __name__ == "__main__":
    main()