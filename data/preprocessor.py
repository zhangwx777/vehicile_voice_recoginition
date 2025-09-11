# data/preprocessor.py

import numpy as np
import librosa
import warnings
import os
from core.logger import logger

# 数据增强功能已移除
AudioAugmentation = None

warnings.filterwarnings('ignore')


class AudioPreprocessor:
    """音频预处理类"""

    def __init__(self, sample_rate=16000, n_mels=128, n_fft=2048, hop_length=512, 
                 extract_features=['mel']):
        # 验证参数有效性
        if sample_rate <= 0:
            raise ValueError(f"采样率必须大于0，当前值：{sample_rate}")
        if n_fft <= 0:
            raise ValueError(f"n_fft必须大于0，当前值：{n_fft}")
        if hop_length <= 0 or hop_length >= n_fft:
            raise ValueError(f"hop_length必须在(0, {n_fft})范围内，当前值：{hop_length}")
        if n_mels <= 0:
            raise ValueError(f"n_mels必须大于0，当前值：{n_mels}")
            
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        
        # 特征提取配置
        self.extract_features = extract_features if isinstance(extract_features, list) else [extract_features]
        
        # 音频预处理器初始化完成

    def load_audio(self, file_path, duration=None):
        """加载音频文件"""
        # 输入验证
        if not file_path:
            logger.error("音频文件路径不能为空")
            return None
            
        if not os.path.exists(file_path):
            logger.error(f"音频文件不存在：{file_path}")
            return None
            
        # 检查文件格式
        valid_extensions = ['.wav', '.mp3', '.flac', '.ogg', '.m4a']
        file_ext = os.path.splitext(file_path)[1].lower()
        if file_ext not in valid_extensions:
            logger.warning(f"不支持的音频格式：{file_ext}，支持的格式：{valid_extensions}")
        
        # 使用传入的duration或默认值
        actual_duration = duration if duration is not None else getattr(self, 'duration', 4)
        
        if actual_duration <= 0:
            logger.error(f"音频时长必须为正数：{actual_duration}")
            return None
        
        try:
            # 加载音频文件
            audio, sr = librosa.load(file_path, sr=self.sample_rate, duration=actual_duration)
            
            # 检查加载结果
            if audio is None or len(audio) == 0:
                logger.error(f"音频文件加载失败或为空：{file_path}")
                return None
                
            # 检查采样率
            if sr != self.sample_rate:
                logger.warning(f"音频采样率不匹配：期望{self.sample_rate}，实际{sr}")
            
            # 如果音频长度不足，进行填充
            expected_length = self.sample_rate * actual_duration
            if len(audio) < expected_length:
                pad_length = expected_length - len(audio)
                audio = np.pad(audio, (0, pad_length), mode='constant', constant_values=0)
                # 音频文件填充
            elif len(audio) > expected_length:
                # 如果音频过长，截取前面部分
                audio = audio[:expected_length]
                # 音频文件截取
                
            # 检查音频是否全为静音
            if np.max(np.abs(audio)) < 1e-6:
                logger.warning(f"音频文件可能为静音：{file_path}")
                
            return audio
            
        except (OSError, IOError) as e:
            logger.error(f"音频文件读取错误 {file_path}: {str(e)}")
            return None
        except librosa.LibrosaError as e:
            logger.error(f"音频解码错误 {file_path}: {str(e)}")
            return None
        except (ValueError, TypeError) as e:
            logger.error(f"音频参数错误 {file_path}: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"加载音频文件未知错误 {file_path}: {str(e)}")
            return None

    def extract_mel_spectrogram(self, audio):
        """提取梅尔频谱特征"""
        try:
            # 输入验证
            if audio is None:
                logger.error("音频数据为空")
                return None
                
            if len(audio) == 0:
                logger.error("音频数据长度为0")
                return None
                
            # 计算梅尔频谱
            mel_spec = librosa.feature.melspectrogram(
                y=audio,
                sr=self.sample_rate,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                n_mels=self.n_mels
            )
            
            # 检查输出有效性
            if mel_spec is None or mel_spec.size == 0:
                logger.error("梅尔频谱计算失败")
                return None
                
            # 转换为对数刻度
            log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
            
            # 检查是否存在NaN或无穷大值
            if np.any(np.isnan(log_mel_spec)) or np.any(np.isinf(log_mel_spec)):
                logger.warning("梅尔频谱中存在NaN或无穷大值，进行清理")
                log_mel_spec = np.nan_to_num(log_mel_spec, nan=0.0, posinf=0.0, neginf=-80.0)
            
            return log_mel_spec
            
        except Exception as e:
            logger.error(f"提取梅尔频谱失败: {str(e)}")
            return None

    def extract_mfcc(self, audio):
        """提取MFCC特征"""
        try:
            # 提取MFCC特征
            mfcc = librosa.feature.mfcc(
                y=audio,
                sr=self.sample_rate,
                n_mfcc=13,  # 传统上使用13维MFCC
                n_fft=self.n_fft,
                hop_length=self.hop_length
            )
            
            # 检查输出有效性
            if mfcc is None or mfcc.size == 0:
                logger.error("MFCC特征计算失败")
                return None
                
            # 检查NaN和无穷大值
            if np.any(np.isnan(mfcc)) or np.any(np.isinf(mfcc)):
                logger.warning("MFCC特征中存在NaN或无穷大值")
                mfcc = np.nan_to_num(mfcc, nan=0.0, posinf=0.0, neginf=0.0)
            
            return mfcc
            
        except Exception as e:
            logger.error(f"提取MFCC特征失败: {str(e)}")
            return None
    
    def extract_chroma(self, audio):
        """提取色度特征"""
        try:
            # 提取色度特征
            chroma = librosa.feature.chroma_stft(
                y=audio,
                sr=self.sample_rate,
                n_fft=self.n_fft,
                hop_length=self.hop_length
            )
            
            # 检查输出有效性
            if chroma is None or chroma.size == 0:
                logger.error("色度特征计算失败")
                return None
                
            # 检查NaN和无穷大值
            if np.any(np.isnan(chroma)) or np.any(np.isinf(chroma)):
                logger.warning("色度特征中存在NaN或无穷大值")
                chroma = np.nan_to_num(chroma, nan=0.0, posinf=0.0, neginf=0.0)
            
            return chroma
            
        except Exception as e:
            logger.error(f"提取色度特征失败: {str(e)}")
            return None
    
    def extract_spectral_contrast(self, audio):
        """提取频谱对比度特征"""
        try:
            # 提取频谱对比度特征
            spectral_contrast = librosa.feature.spectral_contrast(
                y=audio,
                sr=self.sample_rate,
                n_fft=self.n_fft,
                hop_length=self.hop_length
            )
            
            # 检查输出有效性
            if spectral_contrast is None or spectral_contrast.size == 0:
                logger.error("频谱对比度特征计算失败")
                return None
                
            # 检查NaN和无穷大值
            if np.any(np.isnan(spectral_contrast)) or np.any(np.isinf(spectral_contrast)):
                logger.warning("频谱对比度特征中存在NaN或无穷大值")
                spectral_contrast = np.nan_to_num(spectral_contrast, nan=0.0, posinf=0.0, neginf=0.0)
            
            return spectral_contrast
            
        except Exception as e:
            logger.error(f"提取频谱对比度特征失败: {str(e)}")
            return None

    def preprocess_audio(self, file_path, duration=4):
        """完整的音频预处理流程"""
        audio = self.load_audio(file_path, duration)
        if audio is None:
            return None

        # 数据增强功能已移除

        # 调试信息：打印特征提取配置
        # 特征提取配置
        
        # 提取多种特征
        features_dict = {}
        
        for feature_type in self.extract_features:
            if feature_type == 'mel':
                features = self.extract_mel_spectrogram(audio)
            elif feature_type == 'mfcc':
                features = self.extract_mfcc(audio)
            elif feature_type == 'chroma':
                features = self.extract_chroma(audio)
            elif feature_type == 'spectral_contrast':
                features = self.extract_spectral_contrast(audio)
            else:
                logger.warning(f"不支持的特征类型: {feature_type}")
                continue
                
            if features is not None:
                # 数据增强功能已移除
                        
                features_dict[feature_type] = features
        
        # 如果只有一种特征，直接返回
        if len(features_dict) == 1:
            result = list(features_dict.values())[0]
            # 单一特征形状
            return result
        
        # 如果有多种特征，进行融合
        if len(features_dict) > 1:
            # 检测到多种特征，进行融合
            result = self._fuse_features(features_dict)
            # 融合后特征形状
            return result
        
        # 如果没有成功提取任何特征
        logger.error(f"无法提取任何特征: {file_path}")
        return None
    
    def _fuse_features(self, features_dict):
        """融合多种特征"""
        try:
            # 简单的特征融合：垂直连接
            features_list = []
            min_time_steps = float('inf')
            
            # 找到最小的时间步数
            for features in features_dict.values():
                min_time_steps = min(min_time_steps, features.shape[1])
            
            # 裁剪所有特征到相同的时间步数
            for feature_type, features in features_dict.items():
                features_cropped = features[:, :min_time_steps]
                features_list.append(features_cropped)
                # 特征形状
            
            # 垂直连接特征
            fused_features = np.vstack(features_list)
            # 融合后特征形状
            
            return fused_features
            
        except Exception as e:
            logger.error(f"特征融合失败: {str(e)}")
            # 如果融合失败，返回第一个特征
            return list(features_dict.values())[0]