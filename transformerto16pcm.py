import os
import librosa
import soundfile as sf
import numpy as np
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class AudioConverter:
    """音频格式转换器 - 转换为16bit PCM WAV"""
    
    def __init__(self, target_sr=16000, target_channels=1):
        """
        初始化转换器
        
        Args:
            target_sr: 目标采样率 (默认16kHz, 也可选择44100Hz)
            target_channels: 目标声道数 (1=单声道, 2=立体声)
        """
        self.target_sr = target_sr
        self.target_channels = target_channels
        
    def convert_single_file(self, input_path, output_path=None, 
                           min_duration=0.1, max_duration=None):
        """
        转换单个音频文件
        
        Args:
            input_path: 输入文件路径
            output_path: 输出文件路径 (如果为None则自动生成)
            min_duration: 最小时长(秒)，不足则填充
            max_duration: 最大时长(秒)，超出则截断
        """
        input_path = Path(input_path)
        
        # 生成输出路径
        if output_path is None:
            output_path = input_path.parent / f"{input_path.stem}_16bit.wav"
        else:
            output_path = Path(output_path)
        
        try:
            # 加载音频文件
            audio, original_sr = librosa.load(input_path, sr=None, mono=False)
            original_duration = len(audio) / original_sr if audio.ndim == 1 else len(audio[0]) / original_sr
            
            print(f"原始文件信息:")
            print(f"  采样率: {original_sr} Hz")
            print(f"  时长: {original_duration:.3f} 秒")
            print(f"  声道: {'单声道' if audio.ndim == 1 else f'{audio.shape[0]}声道'}")
            print(f"  数据类型: {audio.dtype}")
            
            # 处理声道
            if audio.ndim > 1:
                if self.target_channels == 1:
                    # 转为单声道
                    audio = librosa.to_mono(audio)
                elif self.target_channels == 2 and audio.shape[0] == 1:
                    # 单声道转立体声
                    audio = np.repeat(audio, 2, axis=0)
            
            # 重采样
            if original_sr != self.target_sr:
                audio = librosa.resample(audio, orig_sr=original_sr, target_sr=self.target_sr)
                print(f"  重采样: {original_sr} Hz → {self.target_sr} Hz")
            
            # 处理音频长度
            current_duration = len(audio) / self.target_sr
            
            # 最小长度检查和填充
            if min_duration and current_duration < min_duration:
                target_samples = int(min_duration * self.target_sr)
                padding = target_samples - len(audio)
                audio = np.pad(audio, (0, padding), mode='constant', constant_values=0)
                print(f"  填充: {current_duration:.3f}秒 → {min_duration:.3f}秒")
                current_duration = min_duration
            
            # 最大长度检查和截断
            if max_duration and current_duration > max_duration:
                target_samples = int(max_duration * self.target_sr)
                audio = audio[:target_samples]
                print(f"  截断: {current_duration:.3f}秒 → {max_duration:.3f}秒")
                current_duration = max_duration
            
            # 归一化到合适的范围 (-1, 1)
            if np.max(np.abs(audio)) > 0:
                # 防止削波，留一点余量
                max_val = np.max(np.abs(audio))
                if max_val > 0.95:
                    audio = audio * (0.95 / max_val)
                    print(f"  音量归一化: {max_val:.3f} → 0.95")
            
            # 保存为16bit PCM WAV
            sf.write(
                output_path, 
                audio, 
                self.target_sr,
                subtype='PCM_16',  # 16bit PCM
                format='WAV'
            )
            
            # 验证输出文件
            verify_audio, verify_sr = sf.read(output_path)
            verify_info = sf.info(output_path)
            
            print(f"转换完成:")
            print(f"  输出文件: {output_path}")
            print(f"  采样率: {verify_sr} Hz")
            print(f"  时长: {len(verify_audio)/verify_sr:.3f} 秒")
            print(f"  格式: {verify_info.format}")
            print(f"  子类型: {verify_info.subtype}")
            print(f"  声道: {verify_info.channels}")
            print(f"  文件大小: {output_path.stat().st_size:,} 字节")
            
            return True, output_path
            
        except Exception as e:
            print(f"转换失败: {str(e)}")
            return False, None
    
    def batch_convert(self, input_dir, output_dir=None, 
                     file_extensions=None, min_duration=0.1, max_duration=None):
        """
        批量转换目录下的音频文件
        
        Args:
            input_dir: 输入目录
            output_dir: 输出目录 (如果为None则在输入目录创建converted子目录)
            file_extensions: 支持的文件扩展名列表
            min_duration: 最小时长(秒)
            max_duration: 最大时长(秒)
        """
        input_dir = Path(input_dir)
        
        if not input_dir.exists():
            print(f"错误: 输入目录 {input_dir} 不存在!")
            return
        
        # 设置输出目录
        if output_dir is None:
            output_dir = input_dir / "converted_16bit"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(exist_ok=True)
        
        # 支持的文件格式
        if file_extensions is None:
            file_extensions = ['.wav', '.mp3', '.flac', '.m4a', '.aac', '.ogg', '.wma']
        
        # 查找所有音频文件
        audio_files = []
        for ext in file_extensions:
            audio_files.extend(input_dir.glob(f"**/*{ext}"))
            audio_files.extend(input_dir.glob(f"**/*{ext.upper()}"))
        
        if not audio_files:
            print(f"在 {input_dir} 中没有找到支持的音频文件!")
            print(f"支持的格式: {file_extensions}")
            return
        
        print(f"找到 {len(audio_files)} 个音频文件")
        print(f"输出目录: {output_dir}")
        print("=" * 60)
        
        success_count = 0
        failed_files = []
        
        # 批量处理
        for audio_file in tqdm(audio_files, desc="转换音频文件"):
            # 生成输出文件名
            output_file = output_dir / f"{audio_file.stem}_16bit.wav"
            
            print(f"\n处理: {audio_file.name}")
            success, output_path = self.convert_single_file(
                audio_file, 
                output_file,
                min_duration=min_duration,
                max_duration=max_duration
            )
            
            if success:
                success_count += 1
            else:
                failed_files.append(audio_file.name)
        
        # 输出统计结果
        print("\n" + "=" * 60)
        print(f"批量转换完成!")
        print(f"成功转换: {success_count}/{len(audio_files)} 个文件")
        
        if failed_files:
            print(f"失败文件: {len(failed_files)} 个")
            for failed_file in failed_files:
                print(f"  - {failed_file}")
        
        print(f"输出目录: {output_dir}")

def inspect_audio_files(directory):
    """检查目录下音频文件的详细信息"""
    directory = Path(directory)
    
    if not directory.exists():
        print(f"错误: 目录 {directory} 不存在!")
        return
    
    # 查找音频文件
    audio_extensions = ['.wav', '.mp3', '.flac', '.m4a', '.aac', '.ogg', '.wma']
    audio_files = []
    
    for ext in audio_extensions:
        audio_files.extend(directory.glob(f"**/*{ext}"))
        audio_files.extend(directory.glob(f"**/*{ext.upper()}"))
    
    if not audio_files:
        print(f"在 {directory} 中没有找到音频文件!")
        return
    
    print(f"音频文件检查报告 - 共找到 {len(audio_files)} 个文件")
    print("=" * 80)
    
    for audio_file in audio_files:
        try:
            # 获取文件信息
            file_info = sf.info(audio_file)
            file_size = audio_file.stat().st_size
            
            print(f"文件: {audio_file.name}")
            print(f"  路径: {audio_file}")
            print(f"  大小: {file_size:,} 字节 ({file_size/1024:.1f} KB)")
            print(f"  格式: {file_info.format}")
            print(f"  子类型: {file_info.subtype}")
            print(f"  采样率: {file_info.samplerate} Hz")
            print(f"  声道数: {file_info.channels}")
            print(f"  时长: {file_info.duration:.3f} 秒")
            print(f"  帧数: {file_info.frames:,}")
            
            # 检查潜在问题
            issues = []
            if file_info.duration < 0.1:
                issues.append("⚠️ 音频太短")
            if file_info.samplerate < 8000:
                issues.append("⚠️ 采样率过低")
            if file_info.subtype not in ['PCM_16', 'PCM_24', 'PCM_32']:
                issues.append(f"⚠️ 非PCM格式 ({file_info.subtype})")
            if file_size < 1000:
                issues.append("⚠️ 文件过小")
            
            if issues:
                print(f"  问题: {', '.join(issues)}")
            else:
                print(f"  状态: ✅ 正常")
                
        except Exception as e:
            print(f"文件: {audio_file.name}")
            print(f"  ❌ 读取失败: {str(e)}")
        
        print("-" * 40)

if __name__ == "__main__":
    # 设置路径
    input_directory = r"D:\Car\vehicile_voice_recoginition\data"
    
    print("音频格式转换器 - 转换为16bit PCM WAV")
    print("=" * 50)
    
    # 1. 检查现有音频文件
    print("1. 检查现有音频文件...")
    inspect_audio_files(input_directory)
    
    # 2. 询问是否进行转换
    print("\n" + "=" * 50)
    response = input("是否将所有音频文件转换为16bit PCM WAV格式? (y/n): ")
    
    if response.lower() == 'y':
        # 3. 执行批量转换
        converter = AudioConverter(
            target_sr=16000,  # 16kHz采样率（适合语音识别）
            target_channels=1  # 单声道
        )
        
        converter.batch_convert(
            input_dir=input_directory,
            output_dir=None,  # 自动创建converted_16bit目录
            min_duration=0.1,  # 最小100ms
            max_duration=None   # 不限制最大时长
        )
        
        print("\n" + "=" * 50)
        print("转换完成! 请查看converted_16bit目录中的转换结果。")
        
    else:
        print("操作已取消。")