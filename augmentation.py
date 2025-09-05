# data/augmentation.py

import numpy as np
import librosa
import torch
import random
from logger import logger


class AudioAugmentation:
    """音频数据增强类"""
    
    def __init__(self, config=None):
        """
        初始化音频数据增强器
        
        Args:
            config: 增强配置字典，包含各种增强方法的参数
        """
        self.config = config or {}
        self.enabled = self.config.get('enabled', True)
        
        # 增强方法配置
        self.time_stretch_config = self.config.get('time_stretch', {
            'enabled': True,
            'rate_range': (0.8, 1.2),
            'probability': 0.3
        })
        
        self.pitch_shift_config = self.config.get('pitch_shift', {
            'enabled': True,
            'steps_range': (-2, 2),
            'probability': 0.3
        })
        
        self.noise_injection_config = self.config.get('noise_injection', {
            'enabled': True,
            'noise_factor_range': (0.01, 0.05),
            'probability': 0.4
        })
        
        self.time_mask_config = self.config.get('time_mask', {
            'enabled': True,
            'max_mask_pct': 0.1,
            'num_masks': 2,
            'probability': 0.3
        })
        
        self.freq_mask_config = self.config.get('freq_mask', {
            'enabled': True,
            'max_mask_pct': 0.1,
            'num_masks': 2,
            'probability': 0.3
        })
        
        logger.info(f"音频数据增强器初始化完成，启用状态: {self.enabled}")
    
    def time_stretch(self, audio, sr):
        """时间拉伸"""
        if not self.time_stretch_config.get('enabled', True):
            return audio
            
        if random.random() > self.time_stretch_config.get('probability', 0.3):
            return audio
            
        try:
            rate_range = self.time_stretch_config.get('rate_range', (0.8, 1.2))
            rate = random.uniform(rate_range[0], rate_range[1])
            
            # 使用librosa进行时间拉伸
            stretched = librosa.effects.time_stretch(audio, rate=rate)
            
            # 确保音频长度一致
            if len(stretched) > len(audio):
                stretched = stretched[:len(audio)]
            elif len(stretched) < len(audio):
                # 用零填充
                stretched = np.pad(stretched, (0, len(audio) - len(stretched)))
                
            return stretched
            
        except Exception as e:
            logger.warning(f"时间拉伸失败: {str(e)}")
            return audio
    
    def pitch_shift(self, audio, sr):
        """音调变换"""
        if not self.pitch_shift_config.get('enabled', True):
            return audio
            
        if random.random() > self.pitch_shift_config.get('probability', 0.3):
            return audio
            
        try:
            steps_range = self.pitch_shift_config.get('steps_range', (-2, 2))
            n_steps = random.uniform(steps_range[0], steps_range[1])
            
            # 使用librosa进行音调变换
            shifted = librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)
            
            return shifted
            
        except Exception as e:
            logger.warning(f"音调变换失败: {str(e)}")
            return audio
    
    def add_noise(self, audio):
        """添加噪声"""
        if not self.noise_injection_config.get('enabled', True):
            return audio
            
        if random.random() > self.noise_injection_config.get('probability', 0.4):
            return audio
            
        try:
            factor_range = self.noise_injection_config.get('noise_factor_range', (0.01, 0.05))
            noise_factor = random.uniform(factor_range[0], factor_range[1])
            
            # 生成高斯噪声
            noise = np.random.normal(0, noise_factor, audio.shape)
            
            # 添加噪声并确保音频范围在[-1, 1]
            noisy_audio = audio + noise
            noisy_audio = np.clip(noisy_audio, -1.0, 1.0)
            
            return noisy_audio
            
        except Exception as e:
            logger.warning(f"噪声注入失败: {str(e)}")
            return audio
    
    def time_mask(self, spectrogram):
        """时间掩码（应用于频谱图）"""
        if not self.time_mask_config.get('enabled', True):
            return spectrogram
            
        if random.random() > self.time_mask_config.get('probability', 0.3):
            return spectrogram
            
        try:
            spec_copy = spectrogram.copy()
            max_mask_pct = self.time_mask_config.get('max_mask_pct', 0.1)
            num_masks = self.time_mask_config.get('num_masks', 2)
            
            _, time_steps = spec_copy.shape
            max_mask_length = int(max_mask_pct * time_steps)
            
            for _ in range(num_masks):
                if max_mask_length > 0:
                    mask_length = random.randint(1, max_mask_length)
                    mask_start = random.randint(0, time_steps - mask_length)
                    spec_copy[:, mask_start:mask_start + mask_length] = 0
            
            return spec_copy
            
        except Exception as e:
            logger.warning(f"时间掩码失败: {str(e)}")
            return spectrogram
    
    def freq_mask(self, spectrogram):
        """频率掩码（应用于频谱图）"""
        if not self.freq_mask_config.get('enabled', True):
            return spectrogram
            
        if random.random() > self.freq_mask_config.get('probability', 0.3):
            return spectrogram
            
        try:
            spec_copy = spectrogram.copy()
            max_mask_pct = self.freq_mask_config.get('max_mask_pct', 0.1)
            num_masks = self.freq_mask_config.get('num_masks', 2)
            
            freq_bins, _ = spec_copy.shape
            max_mask_length = int(max_mask_pct * freq_bins)
            
            for _ in range(num_masks):
                if max_mask_length > 0:
                    mask_length = random.randint(1, max_mask_length)
                    mask_start = random.randint(0, freq_bins - mask_length)
                    spec_copy[mask_start:mask_start + mask_length, :] = 0
            
            return spec_copy
            
        except Exception as e:
            logger.warning(f"频率掩码失败: {str(e)}")
            return spectrogram
    
    def augment_audio(self, audio, sr):
        """对音频应用增强"""
        if not self.enabled:
            return audio
            
        # 应用音频级别的增强
        audio = self.time_stretch(audio, sr)
        audio = self.pitch_shift(audio, sr)
        audio = self.add_noise(audio)
        
        return audio
    
    def augment_spectrogram(self, spectrogram):
        """对频谱图应用增强"""
        if not self.enabled:
            return spectrogram
            
        # 应用频谱图级别的增强
        spectrogram = self.time_mask(spectrogram)
        spectrogram = self.freq_mask(spectrogram)
        
        return spectrogram
    
    def __call__(self, audio=None, spectrogram=None, sr=None):
        """
        统一的增强接口
        
        Args:
            audio: 音频信号（可选）
            spectrogram: 频谱图（可选）
            sr: 采样率（音频增强时需要）
            
        Returns:
            增强后的音频或频谱图
        """
        if audio is not None:
            return self.augment_audio(audio, sr)
        elif spectrogram is not None:
            return self.augment_spectrogram(spectrogram)
        else:
            raise ValueError("必须提供音频或频谱图中的一个")


class TensorAugmentation:
    """基于Tensor的数据增强（用于训练期间）"""
    
    def __init__(self, config=None):
        self.config = config or {}
        self.enabled = self.config.get('enabled', True)
    
    def __call__(self, tensor):
        """对输入tensor应用增强"""
        if not self.enabled:
            return tensor
            
        # 这里可以添加基于tensor的增强方法
        # 例如：随机旋转、翻转等
        
        return tensor