# inference/recognizer.py

import torch
import json
import os
from data.preprocessor import AudioPreprocessor
from core.model import CNNModel
from core.settings import MODEL_CONFIG, AUDIO_CONFIG, PATH_CONFIG
from core.logger import logger
from utils.visualization import plot_audio_features
from utils.visualization import plot_inference_results


class VehicleRecognizer:
    """车辆识别器（用于推理）"""

    def __init__(self, model_path, label_mapping_path, device=None):
        # 输入验证
        if not model_path:
            raise ValueError("模型文件路径不能为空")
        if not label_mapping_path:
            raise ValueError("标签映射文件路径不能为空")
            
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")

        # 验证文件存在
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not os.path.exists(label_mapping_path):
            raise FileNotFoundError(f"Label mapping file not found: {label_mapping_path}")

        # 加载标签映射
        try:
            with open(label_mapping_path, 'r', encoding='utf-8') as f:
                mapping_data = json.load(f)
            
            if not mapping_data:
                raise ValueError("标签映射为空")
            
            # 处理新的JSON格式
            if 'vehicle_to_label' in mapping_data:
                self.label_mapping = mapping_data['vehicle_to_label']
                self.idx_to_label = mapping_data['label_to_vehicle']
                # 转换字符串键为整数
                self.idx_to_label = {int(k): v for k, v in self.idx_to_label.items()}
            else:
                # 兼容旧格式
                self.label_mapping = mapping_data
                self.idx_to_label = {v: k for k, v in self.label_mapping.items()}
                
            logger.info(f"Loaded label mapping with {len(self.label_mapping)} classes")
        except json.JSONDecodeError as e:
            logger.error(f"JSON解析错误: {str(e)}")
            raise
        except (FileNotFoundError, PermissionError) as e:
            logger.error(f"标签映射文件访问错误: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"加载标签映射未知错误: {str(e)}")
            raise

        # 使用配置文件中的参数初始化预处理器
        # 注意：推理时禁用数据增强以确保预测一致性
        try:
            inference_config = AUDIO_CONFIG.copy()
            inference_config['enable_augmentation'] = False  # 推理时强制禁用数据增强
            self.preprocessor = AudioPreprocessor(**inference_config)
            logger.info("Initialized audio preprocessor with config parameters")
        except (TypeError, ValueError) as e:
            logger.error(f"预处理器参数错误: {str(e)}")
            raise
        except ImportError as e:
            logger.error(f"预处理器模块导入错误: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"初始化预处理器未知错误: {str(e)}")
            raise

        # 使用配置文件中的参数加载模型
        try:
            model_config = MODEL_CONFIG.copy()
            model_config['num_classes'] = len(self.label_mapping)
            self.model = CNNModel(**model_config)
            
            # 加载模型状态
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # 检查是否是完整的检查点文件
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                logger.info(f"加载完整检查点文件")
            else:
                self.model.load_state_dict(checkpoint)
                logger.info(f"加载模型状态字典")
                
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"Loaded model from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise

    def predict(self, audio_path, return_all_probabilities=False):
        """预测单个音频"""
        # 验证文件存在
        if not os.path.exists(audio_path):
            logger.error(f"音频文件不存在: {audio_path}")
            return None, 0.0 if not return_all_probabilities else (None, 0.0, None)

        # 预处理音频
        features = self.preprocessor.preprocess_audio(audio_path)
        if features is None:
            logger.error(f"音频处理失败: {audio_path}")
            return None, 0.0 if not return_all_probabilities else (None, 0.0, None)

        # 转换为Tensor
        try:
            features_tensor = torch.FloatTensor(features).unsqueeze(0).unsqueeze(0).to(self.device)
        except Exception as e:
            logger.error(f"特征转换为tensor失败: {str(e)}")
            return None, 0.0 if not return_all_probabilities else (None, 0.0, None)

        # 预测
        try:
            with torch.no_grad():
                outputs = self.model(features_tensor)
                _, predicted = torch.max(outputs, 1)
                # 计算softmax概率
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence = probabilities[0][predicted].item()
                all_probabilities = probabilities[0].cpu().numpy()
        except Exception as e:
            logger.error(f"预测失败: {str(e)}")
            return None, 0.0 if not return_all_probabilities else (None, 0.0, None)

        # 获取标签
        try:
            predicted_label = self.idx_to_label[predicted.item()]
            logger.info(f"预测结果: {predicted_label}, 置信度: {confidence:.4f} ({audio_path})")
            if return_all_probabilities:
                return predicted_label, confidence, all_probabilities
            return predicted_label, confidence
        except Exception as e:
            logger.error(f"获取预测标签失败: {str(e)}")
            return None, 0.0 if not return_all_probabilities else (None, 0.0, None)

    def visualize_audio_features(self, audio_path, save_path=None, show_mfcc=True, show_chroma=True):
        """可视化音频特征"""
        if not os.path.exists(audio_path):
            logger.error(f"音频文件不存在: {audio_path}")
            return False
            
        try:
            # 生成保存路径（如果未提供）
            if save_path is None:
                audio_name = os.path.splitext(os.path.basename(audio_path))[0]
                save_dir = os.path.join(PATH_CONFIG.get('results_dir', 'results'), 'audio_visualizations')
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f'{audio_name}_features.png')
            
            plot_audio_features(
                audio_path=audio_path,
                preprocessor=self.preprocessor,
                save_path=save_path,
                show_mfcc=show_mfcc,
                show_chroma=show_chroma
            )
            logger.info(f"音频特征可视化已保存到: {save_path}")
            return True
        except Exception as e:
            logger.error(f"音频特征可视化失败: {str(e)}")
            return False
    
    def visualize_prediction(self, audio_path, save_path=None, top_k=3):
        """可视化预测结果"""
        # 先进行预测，获取所有类别的概率
        label, confidence, all_probabilities = self.predict(audio_path, return_all_probabilities=True)
        
        if label is None:
            logger.error(f"预测失败，无法可视化: {audio_path}")
            return False
        
        try:
            # 生成保存路径（如果未提供）
            if save_path is None:
                audio_name = os.path.splitext(os.path.basename(audio_path))[0]
                save_dir = os.path.join(PATH_CONFIG.get('results_dir', 'results'), 'prediction_visualizations')
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f'{audio_name}_prediction.png')
            
            plot_inference_results(
                audio_path=audio_path,
                predictions=all_probabilities,
                label_mapping=self.label_mapping,
                top_k=top_k,
                save_path=save_path
            )
            logger.info(f"预测结果可视化已保存到: {save_path}")
            return True
        except Exception as e:
            logger.error(f"预测结果可视化失败: {str(e)}")
            return False
    
    def analyze_and_visualize(self, audio_path, show_mfcc=True, show_chroma=True, top_k=3):
        """一站式分析和可视化音频特征和预测结果"""
        # 可视化音频特征
        feature_success = self.visualize_audio_features(
            audio_path, show_mfcc=show_mfcc, show_chroma=show_chroma
        )
        
        # 可视化预测结果
        prediction_success = self.visualize_prediction(audio_path, top_k=top_k)
        
        return feature_success and prediction_success