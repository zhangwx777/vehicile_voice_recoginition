# scripts/individual_inference.py

import sys
import os
import argparse
import time
import json
from pathlib import Path
import pandas as pd

from recognizer import VehicleRecognizer
from settings import PATH_CONFIG
from logger import logger

class IndividualInferenceEngine:
    """个体识别推理引擎"""
    
    def __init__(self, model_path=None, label_mapping_path=None):
        # 使用个体识别模型的默认路径
        self.model_path = model_path or "models/individual_best_model.pth"
        self.label_mapping_path = label_mapping_path or "models/individual_label_mapping.json"
        
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
            # 进行预测
            predicted_label, confidence = self.recognizer.predict(audio_path)
            
            inference_time = time.time() - start_time
            
            # 获取个体车辆信息
            individual_id = self.label_to_individual.get(predicted_label, f"unknown_{predicted_label}")
            
            # 解析个体信息
            if '_' in individual_id and not individual_id.startswith('unknown_'):
                vehicle_type, individual_num = individual_id.rsplit('_', 1)
            elif individual_id.startswith('unknown_'):
                # 处理未知标签的情况
                clean_id = individual_id.replace('unknown_', '')
                if '_' in clean_id:
                    vehicle_type, individual_num = clean_id.rsplit('_', 1)
                else:
                    vehicle_type = "unknown"
                    individual_num = clean_id
            else:
                vehicle_type = "unknown"
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
                    from visualization import plot_audio_features, plot_inference_results
                    from preprocessor import AudioPreprocessor
                    from settings import AUDIO_CONFIG
                    import numpy as np
                    
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
                    
                    # 创建预测概率数组
                    predictions = np.zeros(len(self.label_to_individual))
                    if individual_id in self.label_to_individual:
                        pred_idx = self.label_to_individual[individual_id]
                        predictions[pred_idx] = confidence
                    
                    plot_inference_results(
                        audio_path, predictions, 
                        {v: k for k, v in self.label_to_individual.items()},
                        top_k=5, save_path=pred_save_path
                    )
                    
                    logger.info(f"📊 可视化结果已保存")
                    
                except Exception as e:
                    logger.warning(f"可视化生成失败: {e}")
            
            return result
            
        except Exception as e:
            logger.error(f"个体识别预测失败: {e}")
            return None
    
    def predict_batch(self, audio_dir, output_file=None, visualize_top=3):
        """批量预测音频文件的车辆个体"""
        if not os.path.exists(audio_dir):
            logger.error(f"音频目录不存在: {audio_dir}")
            return []
        
        # 获取所有音频文件
        audio_files = []
        for ext in ['*.wav', '*.mp3', '*.flac', '*.m4a']:
            audio_files.extend(Path(audio_dir).rglob(ext))
        
        if not audio_files:
            logger.warning(f"在目录 {audio_dir} 中未找到音频文件")
            return []
        
        logger.info(f"开始批量个体识别，共 {len(audio_files)} 个文件")
        
        results = []
        start_time = time.time()
        
        for i, audio_file in enumerate(audio_files, 1):
            logger.info(f"处理进度: {i}/{len(audio_files)} - {audio_file.name}")
            
            result = self.predict_single_file(str(audio_file))
            if result:
                results.append(result)
        
        total_time = time.time() - start_time
        
        logger.info(f"✅ 批量个体识别完成!")
        logger.info(f"总处理时间: {total_time:.2f}秒")
        logger.info(f"平均每文件: {total_time/len(audio_files):.3f}秒")
        logger.info(f"成功识别: {len(results)}/{len(audio_files)} 个文件")
        
        # 保存结果
        if output_file and results:
            self._save_batch_results(results, output_file)
        
        # 生成统计报告
        if results:
            self._generate_batch_summary(results)
        
        # 可视化top结果
        if visualize_top > 0 and results:
            self._visualize_top_predictions(results, visualize_top)
        
        return results
    
    def _save_batch_results(self, results, output_file):
        """保存批量预测结果"""
        try:
            df = pd.DataFrame(results)
            
            # 保存为CSV
            csv_file = output_file.replace('.json', '.csv') if output_file.endswith('.json') else f"{output_file}.csv"
            df.to_csv(csv_file, index=False, encoding='utf-8')
            
            # 保存为JSON
            json_file = output_file.replace('.csv', '.json') if output_file.endswith('.csv') else f"{output_file}.json"
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            logger.info(f"📄 批量结果已保存:")
            logger.info(f"   CSV: {csv_file}")
            logger.info(f"   JSON: {json_file}")
            
        except Exception as e:
            logger.error(f"保存批量结果失败: {e}")
    
    def _generate_batch_summary(self, results):
        """生成批量预测统计摘要"""
        df = pd.DataFrame(results)
        
        logger.info(f"\n📊 个体识别统计摘要:")
        logger.info(f"总文件数: {len(results)}")
        logger.info(f"平均置信度: {df['confidence'].mean():.4f}")
        logger.info(f"置信度标准差: {df['confidence'].std():.4f}")
        logger.info(f"最高置信度: {df['confidence'].max():.4f}")
        logger.info(f"最低置信度: {df['confidence'].min():.4f}")
        
        # 按车辆类型统计
        logger.info(f"\n🚗 按车辆类型分布:")
        type_counts = df['vehicle_type'].value_counts()
        for vehicle_type, count in type_counts.items():
            percentage = (count / len(results)) * 100
            logger.info(f"   {vehicle_type}: {count} ({percentage:.1f}%)")
        
        # 按个体统计
        logger.info(f"\n🔍 按个体分布:")
        individual_counts = df['predicted_individual'].value_counts()
        for individual, count in individual_counts.head(10).items():
            percentage = (count / len(results)) * 100
            logger.info(f"   {individual}: {count} ({percentage:.1f}%)")
    
    def _visualize_top_predictions(self, results, top_k):
        """可视化置信度最高的预测结果"""
        try:
            # 按置信度排序
            sorted_results = sorted(results, key=lambda x: x['confidence'], reverse=True)
            top_results = sorted_results[:top_k]
            
            logger.info(f"\n🎨 生成置信度最高的 {len(top_results)} 个预测可视化...")
            
            for i, result in enumerate(top_results, 1):
                audio_file = result['audio_file']
                confidence = result['confidence']
                individual = result['predicted_individual']
                
                logger.info(f"   {i}. {audio_file} -> {individual} (置信度: {confidence:.4f})")
                
                # 生成可视化
                try:
                    from visualization import plot_audio_features, plot_inference_results
                    from preprocessor import AudioPreprocessor
                    from settings import AUDIO_CONFIG
                    import numpy as np
                    
                    # 初始化预处理器
                    vis_config = AUDIO_CONFIG.copy()
                    vis_config['enable_augmentation'] = False
                    preprocessor = AudioPreprocessor(**vis_config)
                    
                    # 创建可视化目录
                    top_vis_dir = "results/top_predictions_visualizations"
                    os.makedirs(top_vis_dir, exist_ok=True)
                    
                    audio_filename = os.path.splitext(os.path.basename(audio_file))[0]
                    
                    # 音频特征可视化
                    feature_save_path = os.path.join(top_vis_dir, f"top_{i}_{audio_filename}_features.png")
                    plot_audio_features(
                        audio_file, preprocessor,
                        save_path=feature_save_path,
                        show_mfcc=True, show_chroma=True
                    )
                    
                    # 预测结果可视化
                    pred_save_path = os.path.join(top_vis_dir, f"top_{i}_{audio_filename}_prediction.png")
                    
                    # 创建预测概率数组
                    predictions = np.zeros(len(self.label_to_individual))
                    if individual in self.label_to_individual:
                        pred_idx = self.label_to_individual[individual]
                        predictions[pred_idx] = confidence
                    
                    plot_inference_results(
                        audio_file, predictions,
                        {v: k for k, v in self.label_to_individual.items()},
                        top_k=5, save_path=pred_save_path
                    )
                    
                except Exception as vis_e:
                    logger.warning(f"第{i}个文件可视化失败: {vis_e}")
            
        except Exception as e:
            logger.warning(f"可视化生成失败: {e}")
    
    def interactive_mode(self):
        """交互式个体识别模式"""
        logger.info(f"\n🎤 进入交互式个体识别模式")
        logger.info(f"输入音频文件路径进行个体识别，输入 'quit' 退出")
        
        while True:
            try:
                audio_path = input("\n请输入音频文件路径: ").strip()
                
                if audio_path.lower() in ['quit', 'exit', 'q']:
                    logger.info("退出交互模式")
                    break
                
                if not audio_path:
                    continue
                
                # 移除引号
                audio_path = audio_path.strip('"\'')
                
                result = self.predict_single_file(audio_path, visualize=True)
                
                if result:
                    print(f"\n结果: {result['predicted_individual']} (置信度: {result['confidence']:.4f})")
                
            except KeyboardInterrupt:
                logger.info("\n用户中断，退出交互模式")
                break
            except Exception as e:
                logger.error(f"交互模式错误: {e}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='车辆个体识别推理工具')
    parser.add_argument('--audio', '-a', type=str, help='单个音频文件路径')
    parser.add_argument('--batch', '-b', type=str, help='批量处理音频目录')
    parser.add_argument('--output', '-o', type=str, help='批量处理结果输出文件')
    parser.add_argument('--model', '-m', type=str, help='个体识别模型路径')
    parser.add_argument('--mapping', type=str, help='个体标签映射文件路径')
    parser.add_argument('--visualize', '-v', action='store_true', default=True, help='生成可视化结果（默认启用）')
    parser.add_argument('--no-visualize', action='store_false', dest='visualize', help='禁用可视化结果')
    parser.add_argument('--interactive', '-i', action='store_true', help='交互式模式')
    parser.add_argument('--top', '-t', type=int, default=3, help='可视化top-k结果数量')
    
    args = parser.parse_args()
    
    try:
        # 初始化推理引擎
        engine = IndividualInferenceEngine(
            model_path=args.model,
            label_mapping_path=args.mapping
        )
        
        if args.interactive:
            # 交互式模式
            engine.interactive_mode()
        elif args.audio:
            # 单文件预测
            result = engine.predict_single_file(args.audio, visualize=args.visualize)
            return result is not None
        elif args.batch:
            # 批量预测
            results = engine.predict_batch(
                args.batch, 
                output_file=args.output,
                visualize_top=args.top
            )
            return len(results) > 0
        else:
            # 默认进入交互模式
            engine.interactive_mode()
        
        return True
        
    except Exception as e:
        logger.error(f"个体识别推理失败: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)