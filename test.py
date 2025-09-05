import os
import torch
import librosa
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import warnings
import torchaudio
import torch.nn.functional as F

warnings.filterwarnings('ignore')

# 导入音频特征提取模型
try:
    from torch_vggish_yamnet import yamnet, vggish
    # 不使用有问题的WaveformToInput转换器
    # from torch_vggish_yamnet.input_proc import WaveformToInput
except ImportError:
    print("警告: torch_vggish_yamnet 模块未安装，请安装相关依赖")
    print("pip install torch-vggish-yamnet")
    exit(1)

class AudioFeatureExtractor:
    """音频特征提取器 - 修复版"""
    
    def __init__(self, target_sr=16000):
        self.target_sr = target_sr
        
        # 初始化模型
        print("正在加载预训练模型...")
        try:
            self.embedding_yamnet = yamnet.yamnet(pretrained=True)
            self.embedding_vggish = vggish.get_vggish(with_classifier=False, pretrained=True)
            
            # 设置为评估模式
            self.embedding_yamnet.eval()
            self.embedding_vggish.eval()
            
            print("模型加载完成!")
            
            # 检查模型参数是否正确加载
            yamnet_params = sum(p.numel() for p in self.embedding_yamnet.parameters())
            vggish_params = sum(p.numel() for p in self.embedding_vggish.parameters())
            print(f"YAMNet参数数量: {yamnet_params:,}")
            print(f"VGGish参数数量: {vggish_params:,}")
            
        except Exception as e:
            print(f"模型加载失败: {e}")
            raise
    
    def load_audio(self, wav_path):
        """加载音频文件"""
        try:
            # 检查文件是否存在
            if not Path(wav_path).exists():
                raise FileNotFoundError(f"音频文件不存在: {wav_path}")
            
            # 使用librosa加载音频
            audio, sr = librosa.load(wav_path, sr=self.target_sr, mono=True)
            print(f"加载音频: {Path(wav_path).name}")
            print(f"  - 采样率: {sr} Hz")
            print(f"  - 音频长度: {len(audio)} 样本点 ({len(audio)/sr:.3f}秒)")
            print(f"  - 音频范围: [{audio.min():.6f}, {audio.max():.6f}]")
            
            # 检查音频是否为静音
            if np.abs(audio).max() < 1e-6:
                print("  - 警告: 音频信号非常微弱，可能是静音文件")
                return None, None
            
            # 检查音频长度是否足够
            min_duration = 0.975  # 最小时长（秒）
            if len(audio) / sr < min_duration:
                print(f"  - 警告: 音频时长过短 ({len(audio)/sr:.3f}s < {min_duration}s)")
                # 对于短音频，使用反射填充
                min_samples = int(min_duration * sr)
                if len(audio) < min_samples:
                    pad_length = min_samples - len(audio)
                    audio = np.pad(audio, (0, pad_length), mode='reflect')
                    print(f"  - 音频已填充到 {len(audio)} 样本点")
            
            return torch.from_numpy(audio.astype(np.float32)), sr
            
        except Exception as e:
            print(f"加载音频文件失败 {wav_path}: {str(e)}")
            return None, None
    
    def preprocess_audio_manual(self, waveform, sample_rate):
        """手动实现音频预处理，替代有问题的WaveformToInput"""
        try:
            # 确保音频是正确的形状和类型
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)  # [N] -> [1, N]
            
            print(f"  - 预处理输入形状: {waveform.shape}")
            print(f"  - 预处理输入范围: [{waveform.min():.6f}, {waveform.max():.6f}]")
            
            # 参数设置（基于YAMNet和VGGish的原始实现）
            n_fft = 512
            hop_length = 160  # 10ms at 16kHz
            n_mels = 64
            
            # 确保音频长度足够进行STFT
            min_length = n_fft
            if waveform.shape[-1] < min_length:
                pad_length = min_length - waveform.shape[-1]
                waveform = F.pad(waveform, (0, pad_length), mode='reflect')
                print(f"  - 音频填充到 {waveform.shape[-1]} 样本点")
            
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
            
            print(f"  - Mel spectrogram形状: {log_mel.shape}")
            
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
            # VGGish和YAMNet期望 [batch, channels, height, width] 格式
            patches_tensor = patches_tensor.squeeze(1)  # [num_patches, n_mels, patch_frames]
            patches_tensor = patches_tensor.unsqueeze(1)  # [num_patches, 1, n_mels, patch_frames]
            
            print(f"  - 最终输出形状: {patches_tensor.shape}")
            print(f"  - 最终输出范围: [{patches_tensor.min():.6f}, {patches_tensor.max():.6f}]")
            
            return patches_tensor
            
        except Exception as e:
            print(f"  - 音频预处理失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def extract_features(self, wav_path):
        """提取单个音频文件的特征 - 修复版"""
        print(f"\n{'='*60}")
        print(f"正在处理: {Path(wav_path).name}")
        print(f"{'='*60}")
        
        # 加载音频
        x_in, in_sr = self.load_audio(wav_path)
        if x_in is None:
            return None, None
        
        try:
            # 预处理
            print("正在进行预处理...")
            in_tensor = self.preprocess_audio_manual(x_in, in_sr)
            
            if in_tensor is None:
                print("预处理失败")
                return None, None
            
            print(f"  - 预处理后输入形状: {in_tensor.shape}")
            
            # 检查输入有效性
            if torch.isnan(in_tensor).any() or torch.isinf(in_tensor).any():
                print("  - 警告: 输入包含NaN或Inf值")
                return None, None
            
            # 特征提取
            yamnet_features = None
            vggish_features = None
            
            with torch.no_grad():
                # YAMNet特征提取
                try:
                    print("  - 正在进行YAMNet特征提取...")
                    
                    # YAMNet返回 (embeddings, logits)
                    result = self.embedding_yamnet(in_tensor)
                    
                    if isinstance(result, tuple):
                        emb_yamnet, _ = result  # 取embeddings，忽略logits
                    else:
                        emb_yamnet = result
                        
                    print(f"    YAMNet原始输出形状: {emb_yamnet.shape}")
                    
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
                        
                    print(f"    YAMNet特征形状: {yamnet_features.shape}")
                    print(f"    YAMNet特征范围: [{yamnet_features.min():.6f}, {yamnet_features.max():.6f}]")
                    
                except Exception as e:
                    print(f"    YAMNet特征提取失败: {e}")
                    import traceback
                    print(f"    详细错误: {traceback.format_exc()}")
                    yamnet_features = None
                
                # VGGish特征提取
                try:
                    print("  - 正在进行VGGish特征提取...")
                    
                    emb_vggish = self.embedding_vggish(in_tensor)
                    print(f"    VGGish原始输出形状: {emb_vggish.shape}")
                    
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
                        
                    print(f"    VGGish特征形状: {vggish_features.shape}")
                    print(f"    VGGish特征范围: [{vggish_features.min():.6f}, {vggish_features.max():.6f}]")
                    
                except Exception as e:
                    print(f"    VGGish特征提取失败: {e}")
                    import traceback
                    print(f"    详细错误: {traceback.format_exc()}")
                    vggish_features = None
            
            # 检查特征有效性
            if yamnet_features is not None:
                if np.isnan(yamnet_features).any() or np.isinf(yamnet_features).any():
                    print("  - 警告: YAMNet特征包含NaN或Inf值")
                    yamnet_features = None
                    
            if vggish_features is not None:
                if np.isnan(vggish_features).any() or np.isinf(vggish_features).any():
                    print("  - 警告: VGGish特征包含NaN或Inf值")
                    vggish_features = None
            
            return yamnet_features, vggish_features
            
        except Exception as e:
            print(f"特征提取过程失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return None, None

def validate_environment():
    """验证运行环境"""
    print("验证运行环境...")
    
    # 检查PyTorch
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"GPU数量: {torch.cuda.device_count()}")
    
    # 检查librosa
    print(f"Librosa版本: {librosa.__version__}")
    
    # 检查torchaudio
    try:
        print(f"TorchAudio版本: {torchaudio.__version__}")
    except:
        print("TorchAudio未安装，请安装: pip install torchaudio")
        return False
    
    # 检查必要模块
    try:
        import torch_vggish_yamnet
        print("torch_vggish_yamnet模块: ✓")
        
        # 测试模型加载
        try:
            test_yamnet = yamnet.yamnet(pretrained=True)
            test_vggish = vggish.get_vggish(with_classifier=False, pretrained=True)
            print("模型加载测试: ✓")
            
            # 清理测试模型
            del test_yamnet, test_vggish
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
        except Exception as e:
            print(f"模型加载测试: ✗ ({e})")
            return False
            
    except ImportError as e:
        print(f"torch_vggish_yamnet模块: ✗ (需要安装: {e})")
        return False
    
    return True

def debug_single_file(file_path):
    """调试单个文件"""
    print(f"\n调试模式 - 测试文件: {file_path}")
    print("=" * 80)
    
    # 检查文件基本信息
    if not Path(file_path).exists():
        print(f"错误: 文件 {file_path} 不存在!")
        return False
    
    try:
        # 获取音频基本信息
        duration = librosa.get_duration(filename=file_path)
        y_orig, sr_orig = librosa.load(file_path, sr=None)
        
        print(f"文件信息:")
        print(f"  - 文件大小: {Path(file_path).stat().st_size / 1024:.1f} KB")
        print(f"  - 文件时长: {duration:.3f} 秒")
        print(f"  - 原始采样率: {sr_orig} Hz")
        print(f"  - 原始音频长度: {len(y_orig):,} 样本")
        print(f"  - 原始音频范围: [{y_orig.min():.6f}, {y_orig.max():.6f}]")
        print(f"  - 音频RMS: {np.sqrt(np.mean(y_orig**2)):.6f}")
        
    except Exception as e:
        print(f"无法读取音频文件信息: {e}")
        return False
    
    # 进行特征提取
    try:
        extractor = AudioFeatureExtractor()
        yamnet_feat, vggish_feat = extractor.extract_features(file_path)
        
        if yamnet_feat is not None and vggish_feat is not None:
            print(f"\n{'='*60}")
            print("✓ 特征提取成功!")
            print(f"YAMNet特征:")
            print(f"  - 形状: {yamnet_feat.shape}")
            print(f"  - 范围: [{yamnet_feat.min():.6f}, {yamnet_feat.max():.6f}]")
            print(f"  - 均值: {yamnet_feat.mean():.6f}")
            print(f"  - 标准差: {yamnet_feat.std():.6f}")
            print(f"  - 非零元素: {np.count_nonzero(yamnet_feat)}/{len(yamnet_feat)}")
            
            print(f"VGGish特征:")
            print(f"  - 形状: {vggish_feat.shape}")
            print(f"  - 范围: [{vggish_feat.min():.6f}, {vggish_feat.max():.6f}]")
            print(f"  - 均值: {vggish_feat.mean():.6f}")
            print(f"  - 标准差: {vggish_feat.std():.6f}")
            print(f"  - 非零元素: {np.count_nonzero(vggish_feat)}/{len(vggish_feat)}")
            
            return True
        else:
            print(f"\n{'='*60}")
            print("✗ 特征提取失败!")
            if yamnet_feat is not None:
                print("YAMNet特征提取成功，VGGish特征提取失败")
            elif vggish_feat is not None:
                print("VGGish特征提取成功，YAMNet特征提取失败")
            else:
                print("YAMNet和VGGish特征提取都失败")
            return False
            
    except Exception as e:
        print(f"特征提取器初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def find_test_files(base_dir):
    """查找测试文件"""
    base_path = Path(base_dir)
    
    if not base_path.exists():
        print(f"目录不存在: {base_dir}")
        return []
    
    # 查找音频文件
    audio_extensions = ['*.wav', '*.mp3', '*.flac', '*.m4a', '*.aac']
    audio_files = []
    
    for ext in audio_extensions:
        audio_files.extend(base_path.glob(ext))
        audio_files.extend(base_path.glob(ext.upper()))
    
    # 去重
    unique_files = []
    seen_names = set()
    for f in audio_files:
        if f.name not in seen_names:
            unique_files.append(f)
            seen_names.add(f.name)
    
    return sorted(unique_files)

if __name__ == "__main__":
    # 验证环境
    if not validate_environment():
        print("环境验证失败，请检查依赖安装")
        print("需要安装: pip install torchaudio")
        exit(1)
    
    # 设置测试文件路径
    test_file = r"D:\Car\vehicile_voice_recoginition\data\converted_16bit\1_16bit.wav"
    
    # 如果指定文件不存在，尝试查找其他文件
    if not Path(test_file).exists():
        print(f"指定的测试文件不存在: {test_file}")
        
        search_dirs = [
            r"D:\Car\vehicile_voice_recoginition\data\converted_16bit",
            r"D:\Car\vehicile_voice_recoginition\data",
            r".\data",
            r".\samples",
            r"."
        ]
        
        found_files = []
        for search_dir in search_dirs:
            files = find_test_files(search_dir)
            if files:
                found_files = files
                print(f"在 {search_dir} 找到 {len(files)} 个音频文件")
                break
        
        if found_files:
            print("找到的音频文件:")
            for i, f in enumerate(found_files[:10]):
                print(f"  {i+1}. {f.name} ({f.stat().st_size/1024:.1f} KB)")
            
            test_file = found_files[0]
            print(f"\n使用第一个文件进行调试: {test_file}")
        else:
            print("未找到任何音频文件，请检查路径设置")
            exit(1)
    
    # 运行调试
    success = debug_single_file(test_file)
    
    if success:
        print(f"\n{'='*80}")
        print("🎉 调试完成！特征提取器工作正常。")
    else:
        print(f"\n{'='*80}")
        print("❌ 调试失败！请检查错误信息并修复问题。")