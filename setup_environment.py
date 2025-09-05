# setup_environment.py
# -*- coding: utf-8 -*-
"""
环境设置和验证脚本
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
import sys

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent))

# 导入音频处理测试函数
from tests.test_audio_processing import test_audio_processing

def check_environment():
    """检查运行环境"""
    print("正在检查运行环境...")
    
    # Python版本检查
    python_version = sys.version_info
    print(f"[OK] Python版本: {python_version.major}.{python_version.minor}.{python_version.micro}")
    if python_version < (3, 7):
        print("警告: 建议使用Python 3.7或更高版本")
    
    # PyTorch检查
    print(f"[OK] PyTorch版本: {torch.__version__}")
    
    # CUDA检查
    if torch.cuda.is_available():
        print(f"[OK] CUDA可用: {torch.version.cuda}")
        print(f"[OK] GPU设备: {torch.cuda.get_device_name(0)}")
        print(f"[OK] GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    else:
        print("CUDA不可用，将使用CPU训练")
    
    # 其他库检查
    try:
        import librosa
        print(f"[OK] Librosa版本: {librosa.__version__}")
    except ImportError:
        print("Librosa未安装")
        return False
    
    print(f"[OK] NumPy版本: {np.__version__}")
    
    return True

def setup_directories():
    """创建必要的目录结构"""
    print("\n正在创建目录结构...")
    
    project_root = Path(__file__).parent
    directories = [
        "vehicle_audio_data",
        "models",
        "results",
        "logs",
        "checkpoints",
        "results/audio_visualizations",
        "results/prediction_visualizations",
        "results/training_plots",
        "tests"  # 添加tests目录
    ]
    
    for directory in directories:
        dir_path = project_root / directory
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"[OK] 创建目录: {dir_path}")
    
    return True

def main():
    """主函数"""
    print("车载语音识别系统环境设置")
    print("=" * 50)
    
    # 检查环境
    if not check_environment():
        print("环境检查失败")
        return False
    
    # 设置目录
    if not setup_directories():
        print("目录设置失败")
        return False
    
    # 测试功能
    if not test_audio_processing():
        print("功能测试失败")
        return False
    
    print("\n环境设置完成！系统已准备就绪")
    print("\n下一步操作:")
    print("1. 生成虚拟数据: python generate_enhanced_data.py")
    print("2. 开始训练: python train_monitor.py")
    print("3. 进行推理: python vehicle_inference.py <audio_file>")
    
    return True

if __name__ == "__main__":
    main()