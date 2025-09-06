# utils/individual_visualization.py

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import librosa
import librosa.display
import os
import json
from core.logger import logger
from core.settings import PATH_CONFIG

# 设置matplotlib使用英文字体，避免中文字体缺失警告
plt.rcParams["font.family"] = ["DejaVu Sans", "Arial", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False


def generate_audio_visualizations(data_dir: str = None, max_samples: int = 5):
    """
    生成音频特征可视化
    
    Args:
        data_dir: 个体化数据目录
        max_samples: 每个个体最大可视化样本数
    """
    try:
        # 设置默认路径
        if data_dir is None:
            data_dir = os.path.join(PATH_CONFIG.get('data_dir', 'vehicle_audio_data'), 'individual_vehicles')
        
        if not os.path.exists(data_dir):
            logger.warning(f"个体化数据目录不存在: {data_dir}")
            return False
        
        # 创建可视化输出目录
        vis_dir = os.path.join(PATH_CONFIG.get('results_dir', 'results'), 'individual', 'audio_features')
        os.makedirs(vis_dir, exist_ok=True)
        
        logger.info(f"📁 数据目录: {data_dir}")
        logger.info(f"📁 输出目录: {vis_dir}")
        
        # 导入预处理器
        from data.preprocessor import AudioPreprocessor
        preprocessor = AudioPreprocessor()
        
        total_generated = 0
        
        # 遍历每个个体目录
        for individual_dir in os.listdir(data_dir):
            individual_path = os.path.join(data_dir, individual_dir)
            if not os.path.isdir(individual_path):
                continue
            
            logger.info(f"处理个体: {individual_dir}")
            
            # 获取音频文件列表
            audio_files = [f for f in os.listdir(individual_path) if f.endswith(('.wav', '.mp3', '.flac'))]
            
            # 限制样本数量
            if len(audio_files) > max_samples:
                audio_files = audio_files[:max_samples]
            
            # 为每个音频文件生成特征可视化
            for audio_file in audio_files:
                try:
                    audio_path = os.path.join(individual_path, audio_file)
                    audio_name = os.path.splitext(audio_file)[0]
                    
                    # 生成保存路径
                    save_path = os.path.join(vis_dir, f"{individual_dir}_{audio_name}_features.png")
                    
                    # 生成音频特征可视化
                    from utils.visualization import plot_audio_features
                    plot_audio_features(
                        audio_path, preprocessor, save_path=save_path,
                        show_mfcc=True, show_chroma=True
                    )
                    
                    total_generated += 1
                    
                except Exception as e:
                    logger.warning(f"生成 {audio_file} 特征可视化失败: {e}")
        
        logger.info(f"✅ 音频特征可视化生成完成，共生成 {total_generated} 个文件")
        
        # 内存清理
        import gc
        gc.collect()
        
        return True
        
    except Exception as e:
        logger.error(f"音频特征可视化生成失败: {e}")
        return False


def plot_inference_results(audio_path, predictions, class_names, top_k=5, save_path=None, figsize=(12, 8)):
    """
    绘制推理结果可视化
    
    Args:
        audio_path: 音频文件路径
        predictions: 预测概率数组
        class_names: 类别名称字典 {index: name}
        top_k: 显示前k个预测结果
        save_path: 保存路径
        figsize: 图像大小
    """
    try:
        # 获取top-k预测结果
        top_indices = np.argsort(predictions)[-top_k:][::-1]
        top_probs = predictions[top_indices]
        # 显示完整的个体ID
        top_names = []
        for i in top_indices:
            vehicle_id = class_names.get(i, f'vehicle_{i:02d}')
            # 直接使用完整的个体ID（如sedan_01、suv_02等）
            top_names.append(vehicle_id)
        # 过滤掉置信度为0的预测结果
        valid_predictions = [(name, prob) for name, prob in zip(top_names, top_probs) if prob > 0.001]
        if valid_predictions:
            top_names, top_probs = zip(*valid_predictions)
            top_names, top_probs = list(top_names), list(top_probs)
        else:
            # 如果没有有效预测，至少显示最高的一个
            top_names, top_probs = [top_names[0]], [top_probs[0]]
        
        # 创建图像
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # 设置总标题
        audio_name = os.path.basename(audio_path)
        fig.suptitle(f'Individual Recognition Results: {audio_name}', fontsize=16, y=0.98)
        
        # 左侧：音频波形
        try:
            from data.preprocessor import AudioPreprocessor
            preprocessor = AudioPreprocessor()
            audio = preprocessor.load_audio(audio_path)
            
            librosa.display.waveshow(audio, sr=preprocessor.sample_rate, ax=ax1)
            ax1.set_title('Audio Waveform', fontsize=14)
            ax1.set_xlabel('Time (seconds)', fontsize=12)
            ax1.set_ylabel('Amplitude', fontsize=12)
            ax1.grid(True, linestyle='--', alpha=0.5)
            ax1.set_xlim(0, 3.0)
            ax1.set_xticks([0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
        except Exception as e:
            ax1.text(0.5, 0.5, f'Audio loading failed: {str(e)}', 
                    ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title('Audio Waveform (Failed to Load)', fontsize=14)
        
        # 右侧：预测结果条形图
        colors = plt.cm.viridis(np.linspace(0, 1, len(top_names)))
        bars = ax2.barh(range(len(top_names)), top_probs, color=colors)
        
        ax2.set_yticks(range(len(top_names)))
        ax2.set_yticklabels(top_names, fontsize=11)
        ax2.set_xlabel('Confidence Score', fontsize=12)
        ax2.set_title(f'Top {top_k} Predictions', fontsize=14)
        ax2.set_xlim(0, 1.0)
        
        # 添加数值标签
        for i, (bar, prob) in enumerate(zip(bars, top_probs)):
            ax2.text(prob + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{prob:.3f}', va='center', fontsize=10)
        
        # 高亮最高预测
        if len(bars) > 0:
            bars[0].set_color('#ff7f0e')
            bars[0].set_edgecolor('black')
            bars[0].set_linewidth(2)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
            
    except Exception as e:
        logger.error(f"绘制推理结果失败: {e}")
        if save_path:
            plt.close()


def generate_prediction_visualizations(model_path: str = None, label_mapping_path: str = None, 
                                     data_dir: str = None, max_samples: int = 3):
    """
    生成预测结果可视化
    
    Args:
        model_path: 模型文件路径
        label_mapping_path: 标签映射文件路径
        data_dir: 个体化数据目录
        max_samples: 每个个体最大可视化样本数
    """
    try:
        # 设置默认路径
        if model_path is None:
            model_path = os.path.join(PATH_CONFIG.get('models_dir', 'models'), 'individual_model.pth')
        if label_mapping_path is None:
            label_mapping_path = os.path.join(PATH_CONFIG.get('models_dir', 'models'), 'individual_label_mapping.json')
        if data_dir is None:
            data_dir = os.path.join(PATH_CONFIG.get('data_dir', 'vehicle_audio_data'), 'individual_vehicles')
        
        # 检查文件存在性
        if not os.path.exists(model_path):
            logger.warning(f"模型文件不存在: {model_path}")
            return False
        if not os.path.exists(label_mapping_path):
            logger.warning(f"标签映射文件不存在: {label_mapping_path}")
            return False
        if not os.path.exists(data_dir):
            logger.warning(f"个体化数据目录不存在: {data_dir}")
            return False
        
        # 创建可视化输出目录
        vis_dir = os.path.join(PATH_CONFIG.get('results_dir', 'results'), 'individual', 'predictions')
        os.makedirs(vis_dir, exist_ok=True)
        
        logger.info(f"📁 模型路径: {model_path}")
        logger.info(f"📁 标签映射: {label_mapping_path}")
        logger.info(f"📁 数据目录: {data_dir}")
        logger.info(f"📁 输出目录: {vis_dir}")
        
        # 导入推理引擎
        from individual_inference import IndividualInferenceEngine
        
        # 初始化推理引擎
        engine = IndividualInferenceEngine(
            model_path=model_path,
            label_mapping_path=label_mapping_path
        )
        
        # 加载标签映射
        with open(label_mapping_path, 'r', encoding='utf-8') as f:
            label_mapping = json.load(f)
        
        # 创建反向映射
        idx_to_label = {v: k for k, v in label_mapping.items()}
        
        total_generated = 0
        
        # 遍历每个个体目录
        for individual_dir in os.listdir(data_dir):
            individual_path = os.path.join(data_dir, individual_dir)
            if not os.path.isdir(individual_path):
                continue
            
            logger.info(f"处理个体: {individual_dir}")
            
            # 获取音频文件列表
            audio_files = [f for f in os.listdir(individual_path) if f.endswith(('.wav', '.mp3', '.flac'))]
            
            # 限制样本数量
            if len(audio_files) > max_samples:
                audio_files = audio_files[:max_samples]
            
            # 为每个音频文件生成预测可视化
            for audio_file in audio_files:
                try:
                    audio_path = os.path.join(individual_path, audio_file)
                    audio_name = os.path.splitext(audio_file)[0]
                    
                    # 进行预测
                    result = engine.predict_single_file(audio_path, visualize=False)
                    if result is None:
                        continue
                    
                    # 生成保存路径
                    save_path = os.path.join(vis_dir, f"{individual_dir}_{audio_name}_prediction.png")
                    
                    # 创建预测概率数组
                    predictions = np.zeros(len(label_mapping))
                    predicted_individual = result['predicted_individual']
                    confidence = result['confidence']
                    
                    if predicted_individual in label_mapping:
                        pred_idx = label_mapping[predicted_individual]
                        predictions[pred_idx] = confidence
                    
                    # 生成预测结果可视化
                    plot_inference_results(
                        audio_path, predictions, idx_to_label,
                        top_k=5, save_path=save_path
                    )
                    
                    total_generated += 1
                    
                except Exception as e:
                    logger.warning(f"生成 {audio_file} 预测可视化失败: {e}")
        
        logger.info(f"✅ 预测结果可视化生成完成，共生成 {total_generated} 个文件")
        
        # 内存清理
        import gc
        gc.collect()
        
        return True
        
    except Exception as e:
        logger.error(f"预测结果可视化生成失败: {e}")
        return False


def generate_summary_visualization(model_path: str = None, label_mapping_path: str = None):
    """
    生成训练总结可视化
    
    Args:
        model_path: 模型文件路径
        label_mapping_path: 标签映射文件路径
    """
    try:
        # 设置默认路径
        if label_mapping_path is None:
            label_mapping_path = os.path.join(PATH_CONFIG.get('models_dir', 'models'), 'individual_label_mapping.json')
        
        if not os.path.exists(label_mapping_path):
            logger.warning(f"标签映射文件不存在: {label_mapping_path}")
            return False
        
        # 创建可视化输出目录
        vis_dir = os.path.join(PATH_CONFIG.get('results_dir', 'results'), 'individual')
        os.makedirs(vis_dir, exist_ok=True)
        
        # 加载标签映射
        with open(label_mapping_path, 'r', encoding='utf-8') as f:
            label_mapping = json.load(f)
        
        # 生成个体分布统计图
        plt.figure(figsize=(12, 8))
        individuals = list(label_mapping.keys())
        counts = [1] * len(individuals)  # 假设每个个体有相同数量的样本
        
        plt.bar(range(len(individuals)), counts, color='skyblue', alpha=0.7)
        plt.xlabel('Individual ID', fontsize=12)
        plt.ylabel('Sample Count', fontsize=12)
        plt.title('Individual Vehicle Distribution', fontsize=16)
        plt.xticks(range(len(individuals)), individuals, rotation=45, ha='right')
        plt.grid(True, linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        summary_path = os.path.join(vis_dir, 'individual_summary.png')
        plt.savefig(summary_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✅ 训练总结可视化已保存到: {summary_path}")
        return True
        
    except Exception as e:
        logger.error(f"生成训练总结可视化失败: {e}")
        return False