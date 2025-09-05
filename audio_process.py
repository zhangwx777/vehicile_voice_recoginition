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
from datetime import datetime

warnings.filterwarnings('ignore')

# 导入音频特征提取模型
try:
    from torch_vggish_yamnet import yamnet, vggish
except ImportError:
    print("警告: torch_vggish_yamnet 模块未安装，请安装相关依赖")
    print("pip install torch-vggish-yamnet")
    exit(1)

class BatchAudioFeatureExtractor:
    """批量音频特征提取器"""
    
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
            
            # 检查音频是否为静音
            if np.abs(audio).max() < 1e-6:
                return None, None
            
            # 检查音频长度是否足够
            min_duration = 0.975  # 最小时长（秒）
            if len(audio) / sr < min_duration:
                # 对于短音频，使用反射填充
                min_samples = int(min_duration * sr)
                if len(audio) < min_samples:
                    pad_length = min_samples - len(audio)
                    audio = np.pad(audio, (0, pad_length), mode='reflect')
            
            return torch.from_numpy(audio.astype(np.float32)), sr
            
        except Exception as e:
            print(f"加载音频文件失败 {wav_path}: {str(e)}")
            return None, None
    
    def preprocess_audio_manual(self, waveform, sample_rate):
        """手动实现音频预处理"""
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
            print(f"  - 音频预处理失败: {e}")
            return None
    
    def extract_single_file_features(self, wav_path, verbose=False):
        """提取单个音频文件的特征"""
        if verbose:
            print(f"正在处理: {Path(wav_path).name}")
        
        # 加载音频
        x_in, in_sr = self.load_audio(wav_path)
        if x_in is None:
            return None, None
        
        try:
            # 预处理
            in_tensor = self.preprocess_audio_manual(x_in, in_sr)
            
            if in_tensor is None:
                return None, None
            
            # 检查输入有效性
            if torch.isnan(in_tensor).any() or torch.isinf(in_tensor).any():
                return None, None
            
            # 特征提取
            yamnet_features = None
            vggish_features = None
            
            with torch.no_grad():
                # YAMNet特征提取
                try:
                    result = self.embedding_yamnet(in_tensor)
                    
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
                        
                except Exception as e:
                    if verbose:
                        print(f"    YAMNet特征提取失败: {e}")
                    yamnet_features = None
                
                # VGGish特征提取
                try:
                    emb_vggish = self.embedding_vggish(in_tensor)
                    
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
                        
                except Exception as e:
                    if verbose:
                        print(f"    VGGish特征提取失败: {e}")
                    vggish_features = None
            
            # 检查特征有效性
            if yamnet_features is not None:
                if np.isnan(yamnet_features).any() or np.isinf(yamnet_features).any():
                    yamnet_features = None
                    
            if vggish_features is not None:
                if np.isnan(vggish_features).any() or np.isinf(vggish_features).any():
                    vggish_features = None
            
            return yamnet_features, vggish_features
            
        except Exception as e:
            if verbose:
                print(f"特征提取过程失败: {str(e)}")
            return None, None
    
    def find_audio_files(self, directory):
        """查找目录中的所有音频文件"""
        audio_extensions = ['.wav', '.mp3', '.flac', '.m4a', '.aac', '.ogg', '.wma']
        audio_files = []
        
        directory = Path(directory)
        if not directory.exists():
            print(f"目录不存在: {directory}")
            return []
        
        print(f"正在扫描目录: {directory}")
        
        # 递归查找所有音频文件
        for file_path in directory.rglob('*'):
            if file_path.suffix.lower() in audio_extensions:
                audio_files.append(file_path)
        
        print(f"找到 {len(audio_files)} 个音频文件")
        return sorted(audio_files)
    
    def process_directory(self, input_dir, output_csv_path=None, verbose=True):
        """批量处理目录中的所有音频文件"""
        
        # 查找音频文件
        audio_files = self.find_audio_files(input_dir)
        
        if not audio_files:
            print("未找到任何音频文件")
            return None
        
        # 设置输出文件路径
        if output_csv_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_csv_path = f"audio_features_{timestamp}.csv"
        
        print(f"将处理 {len(audio_files)} 个文件")
        print(f"输出文件: {output_csv_path}")
        
        # 存储结果的列表
        results = []
        failed_files = []
        
        # 使用进度条处理文件
        for audio_file in tqdm(audio_files, desc="提取特征"):
            try:
                yamnet_feat, vggish_feat = self.extract_single_file_features(
                    audio_file, verbose=False
                )
                
                if yamnet_feat is not None and vggish_feat is not None:
                    # 创建特征字典
                    feature_dict = {'filename': audio_file.name}
                    
                    # 添加YAMNet特征
                    for i, feat_val in enumerate(yamnet_feat):
                        feature_dict[f'yamnet_feat_{i:03d}'] = feat_val
                    
                    # 添加VGGish特征
                    for i, feat_val in enumerate(vggish_feat):
                        feature_dict[f'vggish_feat_{i:03d}'] = feat_val
                    
                    results.append(feature_dict)
                    
                    if verbose and len(results) % 10 == 0:
                        print(f"已处理 {len(results)} 个文件...")
                        
                else:
                    failed_files.append(audio_file.name)
                    if verbose:
                        print(f"跳过文件 (特征提取失败): {audio_file.name}")
                    
            except Exception as e:
                failed_files.append(audio_file.name)
                if verbose:
                    print(f"处理文件时出错 {audio_file.name}: {e}")
        
        # 转换为DataFrame并保存
        if results:
            df = pd.DataFrame(results)
            
            # 确保filename列在第一列
            cols = ['filename'] + [col for col in df.columns if col != 'filename']
            df = df[cols]
            
            # 保存为CSV
            df.to_csv(output_csv_path, index=False)
            
            print(f"\n{'='*60}")
            print("处理完成!")
            print(f"成功处理: {len(results)} 个文件")
            print(f"失败文件: {len(failed_files)} 个")
            print(f"输出文件: {output_csv_path}")
            print(f"特征维度: YAMNet={len([c for c in df.columns if 'yamnet' in c])}, "
                  f"VGGish={len([c for c in df.columns if 'vggish' in c])}")
            print(f"CSV文件大小: {Path(output_csv_path).stat().st_size / 1024 / 1024:.2f} MB")
            
            if failed_files and verbose:
                print(f"\n失败的文件:")
                for failed_file in failed_files[:10]:  # 只显示前10个失败文件
                    print(f"  - {failed_file}")
                if len(failed_files) > 10:
                    print(f"  ... 还有 {len(failed_files) - 10} 个失败文件")
            
            return df
        else:
            print("没有成功处理任何文件")
            return None

def validate_environment():
    """验证运行环境"""
    print("验证运行环境...")
    
    # 检查PyTorch
    print(f"PyTorch版本: {torch.__version__}")
    
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
        
        # 简单测试模型加载
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
        print(f"torch_vggish_yamnet模块: ✗ ({e})")
        return False
    
    return True

if __name__ == "__main__":
    # 验证环境
    if not validate_environment():
        print("环境验证失败，请检查依赖安装")
        exit(1)
    
    # 设置输入目录
    input_directory = r"D:\Car\vehicile_voice_recoginition\data"
    
    # 设置输出文件名（可选）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"audio_features_{timestamp}.csv"
    
    print(f"输入目录: {input_directory}")
    print(f"输出文件: {output_file}")
    
    # 检查输入目录是否存在
    if not Path(input_directory).exists():
        print(f"错误: 输入目录不存在: {input_directory}")
        
        # 提供一些建议的路径
        suggested_paths = [
            r"D:\Car\vehicile_voice_recoginition\data\converted_16bit",
            r".\data",
            r".\samples"
        ]
        
        print("建议检查以下路径:")
        for path in suggested_paths:
            if Path(path).exists():
                print(f"  ✓ {path} (存在)")
            else:
                print(f"  ✗ {path} (不存在)")
        
        exit(1)
    
    try:
        # 创建特征提取器并开始处理
        print(f"\n{'='*80}")
        print("开始批量音频特征提取")
        print(f"{'='*80}")
        
        extractor = BatchAudioFeatureExtractor()
        
        # 处理目录
        result_df = extractor.process_directory(
            input_dir=input_directory,
            output_csv_path=output_file,
            verbose=True
        )
        
        if result_df is not None:
            print(f"\n✓ 处理完成！特征已保存到: {output_file}")
            print(f"  数据形状: {result_df.shape}")
            print(f"  前5行预览:")
            print(result_df.head())
        else:
            print("\n✗ 处理失败，未生成输出文件")
            
    except KeyboardInterrupt:
        print("\n\n用户中断处理")
    except Exception as e:
        print(f"\n处理过程中出现错误: {e}")
        import traceback
        traceback.print_exc()