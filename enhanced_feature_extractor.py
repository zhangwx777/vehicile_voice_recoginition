#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强版音频特征提取器
优化YAMNet和VGGish特征提取，支持更好的向量化和特征融合
"""

import os
import torch
import torch.nn as nn
import librosa
import numpy as np
from pathlib import Path
import warnings
import torchaudio
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
import pickle

# 导入ACBlock和CTFA模块
try:
    from core.acblock import ACBlock, MultiScaleACBlock
    from core.ctfa import CTFA, CTFANetwork, MultiHeadCTFA
except ImportError as e:
    print(f"警告: ACBlock/CTFA模块导入失败: {e}")
    ACBlock = None
    MultiScaleACBlock = None
    CTFA = None
    CTFANetwork = None
    MultiHeadCTFA = None

warnings.filterwarnings('ignore')

# 导入音频特征提取模型
try:
    from torch_vggish_yamnet import yamnet, vggish
except ImportError:
    print("警告: torch_vggish_yamnet 模块未安装，请安装相关依赖")
    print("pip install torch-vggish-yamnet")
    exit(1)

class EnhancedAudioFeatureExtractor:
    """增强版音频特征提取器"""
    
    def __init__(self, 
                 target_sr: int = 16000,
                 feature_dim: int = 512,
                 use_pca: bool = True,
                 pca_components: int = 256,
                 normalize_features: bool = True,
                 device: str = 'auto',
                 fusion_weights: Optional[Dict[str, float]] = None,
                 use_acblock: bool = True,
                 use_ctfa: bool = True,
                 **kwargs):
        """
        初始化增强版特征提取器
        
        Args:
            target_sr: 目标采样率
            feature_dim: 最终特征维度
            use_pca: 是否使用PCA降维
            pca_components: PCA组件数量
            normalize_features: 是否标准化特征
            device: 计算设备
            use_acblock: 是否使用ACBlock进行特征筛选
            use_ctfa: 是否使用CTFA进行干扰抑制
        """
        self.target_sr = target_sr
        self.feature_dim = feature_dim
        self.use_pca = use_pca
        self.pca_components = pca_components
        self.normalize_features = normalize_features
        self.use_acblock = use_acblock
        self.use_ctfa = use_ctfa
        
        # 特征融合权重
        self.fusion_weights = fusion_weights or {'yamnet': 0.6, 'vggish': 0.4}
        
        # 设备选择
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # 使用设备: {self.device}
        
        # 初始化模型
        self._load_models()
        
        # 初始化ACBlock和CTFA模块
        self._init_attention_modules()
        
        # 初始化预处理器
        self.scaler = StandardScaler() if normalize_features else None
        self.pca_yamnet = PCA(n_components=pca_components) if use_pca else None
        self.pca_vggish = PCA(n_components=pca_components) if use_pca else None
        
        # 特征统计
        self.feature_stats = {
            'yamnet_mean': None,
            'yamnet_std': None,
            'vggish_mean': None,
            'vggish_std': None
        }
        
    def _load_models(self):
        """加载预训练模型"""
        try:
            # 加载YAMNet（使用正确的方式）
            self.embedding_yamnet = yamnet.yamnet(pretrained=True)
            self.embedding_yamnet.to(self.device)
            self.embedding_yamnet.eval()
            
            # 加载VGGish（使用正确的方式）
            self.embedding_vggish = vggish.get_vggish(with_classifier=False, pretrained=True)
            self.embedding_vggish.to(self.device)
            self.embedding_vggish.eval()
            
        except Exception as e:
            pass  # 模型加载失败时静默处理
            raise
    
    def _init_attention_modules(self):
        """初始化注意力模块"""
        try:
            # 初始化ACBlock模块
            if self.use_acblock and ACBlock is not None:
                self.acblock_network = nn.Sequential(
                    ACBlock(64, 64),
                    ACBlock(64, 64),
                    ACBlock(64, 64)
                ).to(self.device)
                print("✅ ACBlock模块初始化完成")
            else:
                self.acblock_network = None
                print("⚠️  ACBlock模块未启用或不可用")
            
            # 初始化CTFA模块
            if self.use_ctfa and CTFA is not None:
                self.ctfa_network = CTFANetwork(64, num_blocks=2).to(self.device)
                print("✅ CTFA模块初始化完成")
            else:
                self.ctfa_network = None
                print("⚠️  CTFA模块未启用或不可用")
                
        except Exception as e:
            print(f"❌ 注意力模块初始化失败: {e}")
            self.acblock_network = None
            self.ctfa_network = None
    
    def load_audio(self, audio_path: str) -> Tuple[Optional[np.ndarray], Optional[int]]:
        """加载音频文件"""
        try:
            if isinstance(audio_path, (str, Path)):
                # 从文件加载
                audio_path = Path(audio_path)
                if not audio_path.exists():
                    pass  # 音频文件不存在时静默处理
                    return None, None
                
                # 使用librosa加载音频
                waveform, sample_rate = librosa.load(str(audio_path), sr=None)
                
            else:
                # 从内存数据加载
                waveform, sample_rate = audio_path, self.target_sr
            
            # 检查音频有效性
            if waveform is None or len(waveform) == 0:
                pass  # 音频数据为空时静默处理
                return None, None
            
            # 检查音频长度（至少1秒）
            if len(waveform) < sample_rate:
                pass  # 音频太短时静默处理
                return None, None
            
            return waveform, sample_rate
            
        except Exception as e:
            pass  # 音频加载失败时静默处理
            return None, None
    
    def preprocess_audio(self, waveform: np.ndarray, sample_rate: int) -> Optional[torch.Tensor]:
        """音频预处理"""
        try:
            # 重采样到目标采样率
            if sample_rate != self.target_sr:
                waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=self.target_sr)
            
            # 音频增强
            waveform = self._enhance_audio(waveform)
            
            # 确保音频长度适合模型输入
            # YAMNet和VGGish通常需要特定长度的音频
            target_length = self.target_sr * 3  # 3秒
            if len(waveform) > target_length:
                waveform = waveform[:target_length]
            elif len(waveform) < target_length:
                # 零填充
                padded = np.zeros(target_length)
                padded[:len(waveform)] = waveform
                waveform = padded
            
            # 转换为tensor，添加batch维度
            audio_tensor = torch.FloatTensor(waveform).unsqueeze(0).to(self.device)
            
            # 检查有效性
            if torch.isnan(audio_tensor).any() or torch.isinf(audio_tensor).any():
                pass  # 音频包含无效值时静默处理
                return None
            
            return audio_tensor
            
        except Exception as e:
            pass  # 音频预处理失败时静默处理
            return None
    
    def preprocess_audio_manual(self, waveform, sample_rate):
        """手动实现音频预处理（基于原始代码）"""
        try:
            # 确保音频是正确的形状和类型
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)  # [N] -> [1, N]
            
            # 参数设置
            n_fft = 512
            hop_length = 160  # 10ms at 16kHz
            n_mels = 64
            
            # 确保音频长度足够进行STFT
            min_length = n_fft
            if waveform.shape[-1] < min_length:
                pad_length = min_length - waveform.shape[-1]
                waveform = F.pad(waveform, (0, pad_length), mode='reflect')
            
            # 使用torchaudio进行mel spectrogram计算
            mel_spectrogram = torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=n_fft,
                hop_length=hop_length,
                n_mels=n_mels,
                f_min=125.0,
                f_max=7500.0,
                power=2.0,
                normalized=False
            )(waveform)
            
            # 转换为log mel spectrogram
            log_mel = torch.log(mel_spectrogram + 1e-6)
            
            # 创建补丁 (patch)，每个补丁是96帧 (0.96秒)
            patch_frames = 96
            hop_frames = 48  # 50% overlap
            
            patches = []
            for start in range(0, log_mel.shape[-1] - patch_frames + 1, hop_frames):
                patch = log_mel[:, :, start:start + patch_frames]
                if patch.shape[-1] == patch_frames:
                    patches.append(patch)
            
            if not patches:
                # 如果音频太短，创建一个补丁
                if log_mel.shape[-1] < patch_frames:
                    # 填充到所需长度
                    pad_length = patch_frames - log_mel.shape[-1]
                    patch = F.pad(log_mel, (0, pad_length), mode='reflect')
                else:
                    patch = log_mel[:, :, :patch_frames]
                patches = [patch]
            
            # 堆叠所有补丁
            patches_tensor = torch.stack(patches, dim=0)  # [num_patches, 1, n_mels, patch_frames]
            
            # 重新排列维度以匹配模型期望的输入格式
            patches_tensor = patches_tensor.squeeze(1)  # [num_patches, n_mels, patch_frames]
            patches_tensor = patches_tensor.unsqueeze(1)  # [num_patches, 1, n_mels, patch_frames]
            
            return patches_tensor
            
        except Exception as e:
            pass  # 音频预处理失败时静默处理
            return None
    
    def _enhance_audio(self, waveform: np.ndarray) -> np.ndarray:
        """音频增强处理"""
        try:
            # 去除静音
            waveform = librosa.effects.trim(waveform, top_db=20)[0]
            
            # 音量标准化
            if np.max(np.abs(waveform)) > 0:
                waveform = waveform / np.max(np.abs(waveform)) * 0.8
            
            # 高通滤波去除低频噪声
            waveform = librosa.effects.preemphasis(waveform)
            
            return waveform
            
        except Exception as e:
            pass  # 音频增强失败时静默处理
            return waveform
    

    

    
    def fuse_features(self, yamnet_feat: np.ndarray, vggish_feat: np.ndarray) -> np.ndarray:
        """特征融合"""
        try:
            features_list = []
            weights_list = []
            
            # 添加YAMNet特征
            if yamnet_feat is not None:
                features_list.append(yamnet_feat.flatten())
                weights_list.append(self.fusion_weights['yamnet'])
            
            # 添加VGGish特征
            if vggish_feat is not None:
                features_list.append(vggish_feat.flatten())
                weights_list.append(self.fusion_weights['vggish'])
            
            if not features_list:
                return None
            
            # 加权特征融合
            if len(features_list) == 1:
                fused_features = features_list[0]
            else:
                # 确保所有特征具有相同的维度
                min_dim = min(len(feat) for feat in features_list)
                normalized_features = [feat[:min_dim] for feat in features_list]
                
                # 加权平均
                fused_features = np.zeros(min_dim)
                total_weight = sum(weights_list)
                for feat, weight in zip(normalized_features, weights_list):
                    fused_features += feat * (weight / total_weight)
            
            # 简化的特征标准化（避免单样本PCA问题）
            if self.normalize_features:
                # 使用L2标准化替代StandardScaler
                norm = np.linalg.norm(fused_features)
                if norm > 0:
                    fused_features = fused_features / norm
            
            # 只在特征维度确实大于PCA组件数且有足够样本时才使用PCA
            # 对于单样本情况，跳过PCA降维
            if self.use_pca and len(fused_features) > self.pca_components:
                # 检查是否已经拟合过PCA
                if hasattr(self, 'pca_fused') and hasattr(self.pca_fused, 'components_'):
                    try:
                        fused_features = self.pca_fused.transform(fused_features.reshape(1, -1)).flatten()
                    except Exception:
                        # PCA变换失败，跳过PCA
                        pass
                # 对于单样本，不进行PCA拟合
            
            # 调整到目标维度
            if len(fused_features) > self.feature_dim:
                fused_features = fused_features[:self.feature_dim]
            elif len(fused_features) < self.feature_dim:
                # 零填充
                padded_features = np.zeros(self.feature_dim)
                padded_features[:len(fused_features)] = fused_features
                fused_features = padded_features
            
            return fused_features
            
        except Exception as e:
            pass  # 特征融合失败时静默处理
            return None
    
    def extract_features(self, audio_input) -> Optional[Dict[str, Any]]:
        """提取完整的音频特征（使用正确的预处理函数和原始代码逻辑）"""
        try:
            # 加载音频
            waveform, sample_rate = self.load_audio(audio_input)
            if waveform is None:
                return None
            
            # 转换为tensor
            waveform_tensor = torch.FloatTensor(waveform).to(self.device)
            
            # 使用正确的预处理方法
            processed_audio = self.preprocess_audio_manual(waveform_tensor, sample_rate)
            if processed_audio is None:
                return None
            
            # 检查输入有效性
            if torch.isnan(processed_audio).any() or torch.isinf(processed_audio).any():
                return None
            
            features = {}
            
            with torch.no_grad():
                # YAMNet特征提取
                try:
                    result = self.embedding_yamnet(processed_audio)
                    
                    if isinstance(result, tuple):
                        emb_yamnet, _ = result
                    else:
                        emb_yamnet = result
                        
                    # 对时间维度进行平均池化
                    if len(emb_yamnet.shape) > 2:
                        yamnet_features = torch.mean(emb_yamnet, dim=0).cpu().numpy()
                    else:
                        yamnet_features = torch.mean(emb_yamnet, dim=0).cpu().numpy()
                        
                    # 确保特征是1维数组
                    if len(yamnet_features.shape) == 0:
                        yamnet_features = np.array([yamnet_features])
                    elif len(yamnet_features.shape) > 1:
                        yamnet_features = yamnet_features.flatten()
                    
                    # 检查特征有效性
                    if not (np.isnan(yamnet_features).any() or np.isinf(yamnet_features).any()):
                        features['yamnet_features'] = yamnet_features
                        
                except Exception as e:
                    pass  # YAMNet特征提取失败时静默处理
                    features['yamnet_features'] = None
                
                # VGGish特征提取
                try:
                    emb_vggish = self.embedding_vggish(processed_audio)
                    
                    # 对时间维度进行平均池化
                    if len(emb_vggish.shape) > 2:
                        vggish_features = torch.mean(emb_vggish, dim=0).cpu().numpy()
                    else:
                        vggish_features = torch.mean(emb_vggish, dim=0).cpu().numpy()
                        
                    # 确保特征是1维数组
                    if len(vggish_features.shape) == 0:
                        vggish_features = np.array([vggish_features])
                    elif len(vggish_features.shape) > 1:
                        vggish_features = vggish_features.flatten()
                    
                    # 检查特征有效性
                    if not (np.isnan(vggish_features).any() or np.isinf(vggish_features).any()):
                        features['vggish_features'] = vggish_features
                    else:
                        features['vggish_features'] = None
                        
                except Exception as e:
                    pass  # VGGish特征提取失败时静默处理
                    features['vggish_features'] = None
            
            # 应用ACBlock和CTFA特征增强（如果可用）
            enhanced_features = self._apply_attention_enhancement(features)
            
            # 特征融合
            fused_features = self.fuse_features(
                enhanced_features.get('yamnet_features'), 
                enhanced_features.get('vggish_features')
            )
            features['fused_features'] = fused_features
            
            # 返回结果
            result = {
                'yamnet_features': enhanced_features.get('yamnet_features'),
                'vggish_features': enhanced_features.get('vggish_features'),
                'fused_features': fused_features,
                'feature_dim': len(fused_features) if fused_features is not None else 0,
                'audio_duration': len(waveform) / sample_rate,
                'attention_enhanced': self.use_acblock or self.use_ctfa
            }
            
            return result
            
        except Exception as e:
            pass  # 特征提取失败时静默处理
            return None
    
    def _apply_attention_enhancement(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """应用注意力机制增强特征"""
        enhanced_features = features.copy()
        
        try:
            # 对YAMNet特征应用注意力增强
            if (self.use_acblock or self.use_ctfa) and features.get('yamnet_features') is not None:
                yamnet_feat = features['yamnet_features']
                
                # 转换为适合注意力模块的格式
                if len(yamnet_feat.shape) == 1:
                    # 重塑为 [1, 64, 8, 8] 格式（假设512维特征）
                    if len(yamnet_feat) == 512:
                        reshaped_feat = yamnet_feat.reshape(1, 64, 8, 8)
                        feat_tensor = torch.FloatTensor(reshaped_feat).to(self.device)
                        
                        # 应用ACBlock
                        if self.use_acblock and self.acblock_network is not None:
                            feat_tensor = self.acblock_network(feat_tensor)
                        
                        # 应用CTFA
                        if self.use_ctfa and self.ctfa_network is not None:
                            feat_tensor = self.ctfa_network(feat_tensor)
                        
                        # 恢复为1维特征
                        enhanced_yamnet = feat_tensor.cpu().numpy().flatten()
                        enhanced_features['yamnet_features'] = enhanced_yamnet
            
            # 对VGGish特征应用注意力增强
            if (self.use_acblock or self.use_ctfa) and features.get('vggish_features') is not None:
                vggish_feat = features['vggish_features']
                
                # 转换为适合注意力模块的格式
                if len(vggish_feat.shape) == 1:
                    # 重塑为 [1, 64, 8, 8] 格式（假设512维特征）
                    if len(vggish_feat) == 512:
                        reshaped_feat = vggish_feat.reshape(1, 64, 8, 8)
                        feat_tensor = torch.FloatTensor(reshaped_feat).to(self.device)
                        
                        # 应用ACBlock
                        if self.use_acblock and self.acblock_network is not None:
                            feat_tensor = self.acblock_network(feat_tensor)
                        
                        # 应用CTFA
                        if self.use_ctfa and self.ctfa_network is not None:
                            feat_tensor = self.ctfa_network(feat_tensor)
                        
                        # 恢复为1维特征
                        enhanced_vggish = feat_tensor.cpu().numpy().flatten()
                        enhanced_features['vggish_features'] = enhanced_vggish
                        
        except Exception as e:
            print(f"注意力增强失败: {e}")
            # 失败时返回原始特征
            return features
        
        return enhanced_features
    
    def save_preprocessors(self, save_path: str):
        """保存预处理器"""
        try:
            save_data = {
                'scaler': self.scaler,
                'pca_yamnet': self.pca_yamnet,
                'pca_vggish': self.pca_vggish,
                'feature_stats': self.feature_stats,
                'config': {
                    'target_sr': self.target_sr,
                    'feature_dim': self.feature_dim,
                    'use_pca': self.use_pca,
                    'pca_components': self.pca_components,
                    'normalize_features': self.normalize_features
                }
            }
            
            with open(save_path, 'wb') as f:
                pickle.dump(save_data, f)
            
            print(f"预处理器已保存到: {save_path}")
            
        except Exception as e:
            print(f"保存预处理器失败: {e}")
    
    def load_preprocessors(self, load_path: str):
        """加载预处理器"""
        try:
            with open(load_path, 'rb') as f:
                save_data = pickle.load(f)
            
            self.scaler = save_data.get('scaler')
            self.pca_yamnet = save_data.get('pca_yamnet')
            self.pca_vggish = save_data.get('pca_vggish')
            self.feature_stats = save_data.get('feature_stats', {})
            
            print(f"预处理器已从 {load_path} 加载")
            
        except Exception as e:
            print(f"加载预处理器失败: {e}")

def test_enhanced_extractor():
    """测试增强版特征提取器"""
    print("=== 测试增强版特征提取器 ===")
    
    # 创建提取器
    extractor = EnhancedAudioFeatureExtractor(
        feature_dim=512,
        use_pca=True,
        pca_components=256,
        normalize_features=True
    )
    
    # 生成测试音频
    test_audio = np.random.randn(16000 * 3)  # 3秒音频
    
    # 提取特征
    result = extractor.extract_features(test_audio)
    
    if result:
        print(f"特征提取成功!")
        print(f"YAMNet特征维度: {len(result['yamnet_features']) if result['yamnet_features'] is not None else 0}")
        print(f"VGGish特征维度: {len(result['vggish_features']) if result['vggish_features'] is not None else 0}")
        print(f"融合特征维度: {result['feature_dim']}")
        print(f"音频时长: {result['audio_duration']:.2f}秒")
    else:
        print("特征提取失败!")

if __name__ == "__main__":
    test_enhanced_extractor()
