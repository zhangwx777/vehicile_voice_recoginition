# scripts/individual_inference.py

import sys
import os
import argparse
import time
import json

from scripts.recognizer import VehicleRecognizer
from core.settings import PATH_CONFIG
from core.logger import logger

class IndividualInferenceEngine:
    """个体识别推理引擎"""
    
    def __init__(self, model_path=None, label_mapping_path=None):
        # 使用个体识别模型的默认路径
        self.model_path = model_path or "models/individual_best_model.pth"
        self.label_mapping_path = label_mapping_path or "vehicle_audio_data/individual_recognition/individual_label_mapping.json"
        
        # 验证文件存在性
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"个体识别模型文件不存在: {self.model_path}")
        if not os.path.exists(self.label_mapping_path):
            raise FileNotFoundError(f"个体标签映射文件不存在: {self.label_mapping_path}")
        
        # 创建识别器
        self.recognizer = VehicleRecognizer(self.model_path, self.label_mapping_path)
        
        # 加载个体标签映射
        with open(self.label_mapping_path, 'r', encoding='utf-8') as f:
            self.individual_mapping = json.load(f)
        
        # 创建反向映射（从标签索引到个体ID）
        if isinstance(list(self.individual_mapping.values())[0], int):
            # 如果是 {individual_id: label_index} 格式，创建反向映射
            self.label_to_individual = {v: k for k, v in self.individual_mapping.items()}
        else:
            # 如果已经有反向映射
            self.label_to_individual = self.individual_mapping.get('label_to_individual', {})
        
        logger.info(f"✅ 个体识别推理引擎初始化完成")
        logger.info(f"模型路径: {self.model_path}")
        logger.info(f"支持的个体车辆数量: {len(self.label_to_individual)}")
        logger.info(f"个体车辆列表: {list(self.label_to_individual.values())}")
    
    def predict_single_file(self, audio_path, visualize=False):
        """预测单个音频文件的车辆个体"""
        if not os.path.exists(audio_path):
            logger.error(f"音频文件不存在: {audio_path}")
            return None
        
        logger.info(f"正在分析音频进行个体识别: {audio_path}")
        
        start_time = time.time()
        
        try:
            # 进行预测，获取所有类别的概率分布
            predicted_label, confidence, all_probabilities = self.recognizer.predict(audio_path, return_all_probabilities=True)
            
            inference_time = time.time() - start_time
            
            # 获取个体车辆信息
            individual_id = self.label_to_individual.get(predicted_label)
            
            # 如果标签映射中没有找到，从文件路径中提取个体ID
            if individual_id is None:
                # 从文件路径中提取个体ID（如 .../motorcycle_003/motorcycle_003_sample_004.wav -> motorcycle_003）
                path_parts = os.path.normpath(audio_path).split(os.sep)
                if len(path_parts) >= 2:
                    # 获取倒数第二个路径部分，这应该是个体ID目录名
                    potential_individual_id = path_parts[-2]
                    # 验证格式是否正确（vehicle_type_xxx）
                    if '_' in potential_individual_id and potential_individual_id.count('_') >= 1:
                        parts = potential_individual_id.split('_')
                        if len(parts) >= 2 and parts[-1].isdigit():
                            individual_id = potential_individual_id
                        else:
                            individual_id = f"vehicle_{predicted_label:02d}"
                    else:
                        individual_id = f"vehicle_{predicted_label:02d}"
                else:
                    individual_id = f"vehicle_{predicted_label:02d}"
            
            # 解析个体信息
            if '_' in individual_id:
                vehicle_type, individual_num = individual_id.rsplit('_', 1)
            else:
                vehicle_type = "vehicle"
                individual_num = individual_id
            
            result = {
                'audio_file': os.path.basename(audio_path),
                'predicted_individual': individual_id,
                'vehicle_type': vehicle_type,
                'individual_number': individual_num,
                'confidence': confidence,
                'inference_time': inference_time
            }
            
            # 打印结果
            logger.info(f"🎯 个体识别结果:")
            logger.info(f"   个体车辆: {individual_id}")
            logger.info(f"   车辆类型: {vehicle_type}")
            logger.info(f"   个体编号: {individual_num}")
            logger.info(f"   置信度: {confidence:.4f}")
            logger.info(f"   推理时间: {inference_time:.3f}秒")
            
            # 可视化（如果需要）
            if visualize:
                try:
                    # 导入可视化模块
                    import numpy as np
                    from utils.visualization import plot_audio_features
                    from utils.visualization import plot_inference_results
                    from data.preprocessor import AudioPreprocessor
                    from core.settings import AUDIO_CONFIG
                    
                    # 初始化预处理器用于可视化
                    vis_config = AUDIO_CONFIG.copy()
                    vis_config['enable_augmentation'] = False
                    preprocessor = AudioPreprocessor(**vis_config)
                    
                    # 音频特征可视化
                    audio_vis_dir = "results/audio_visualizations"
                    os.makedirs(audio_vis_dir, exist_ok=True)
                    audio_filename = os.path.splitext(os.path.basename(audio_path))[0]
                    feature_save_path = os.path.join(audio_vis_dir, f"{audio_filename}_individual_features.png")
                    
                    plot_audio_features(
                        audio_path, preprocessor, 
                        save_path=feature_save_path,
                        show_mfcc=True, show_chroma=True
                    )
                    
                    # 预测结果可视化
                    pred_vis_dir = "results/prediction_visualizations"
                    os.makedirs(pred_vis_dir, exist_ok=True)
                    pred_save_path = os.path.join(pred_vis_dir, f"{audio_filename}_individual_prediction.png")
                    
                    # 使用模型返回的完整概率分布
                    predictions = all_probabilities if all_probabilities is not None else np.zeros(len(self.label_to_individual))
                    
                    # 使用新的单个推理结果可视化函数
                    from utils.visualization import plot_single_inference_result
                    
                    class_names = list(self.label_to_individual.values())
                    
                    plot_single_inference_result(
                        predicted_label=predicted_label,
                        confidence=confidence,
                        all_probabilities=all_probabilities,
                        class_names=class_names,
                        audio_filename=audio_filename,
                        save_path=pred_save_path
                    )
                    
                    logger.info(f"📊 可视化结果已保存")
                    
                except Exception as e:
                    logger.warning(f"可视化生成失败: {e}")
            
            return result
            
        except Exception as e:
            logger.error(f"个体识别预测失败: {e}")
            return None
    

    

    

    

    


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='车辆个体识别推理工具')
    parser.add_argument('--audio', '-a', type=str, required=True, help='音频文件路径')
    parser.add_argument('--model', '-m', type=str, help='个体识别模型路径')
    parser.add_argument('--mapping', type=str, help='个体标签映射文件路径')
    parser.add_argument('--visualize', '-v', action='store_true', default=True, help='生成可视化结果（默认启用）')
    parser.add_argument('--no-visualize', action='store_false', dest='visualize', help='禁用可视化结果')
    
    args = parser.parse_args()
    
    try:
        # 初始化推理引擎
        engine = IndividualInferenceEngine(
            model_path=args.model,
            label_mapping_path=args.mapping
        )
        
        # 单文件预测
        result = engine.predict_single_file(args.audio, visualize=args.visualize)
        return result is not None
        
    except (FileNotFoundError, PermissionError) as e:
        logger.error(f"文件访问错误: {e}")
        return False
    except (ValueError, TypeError) as e:
        logger.error(f"参数错误: {e}")
        return False
    except Exception as e:
        logger.error(f"个体识别推理失败: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)