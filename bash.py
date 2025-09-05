#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
音频特征提取与相似度匹配API服务器启动脚本
作者: AI Assistant
创建时间: 2025-09-05
"""

import sys
import os
from pathlib import Path

def check_dependencies():
    """检查必要的依赖包"""
    required_packages = [
        'torch',
        'librosa',
        'pandas',
        'numpy',
        'torchaudio',
        'sklearn',
        'fastapi',
        'uvicorn',
        'python-multipart',
        'torch_vggish_yamnet'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'torch_vggish_yamnet':
                import torch_vggish_yamnet
            else:
                __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} - 未安装")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n缺少以下依赖包，请安装:")
        for pkg in missing_packages:
            if pkg == 'torch_vggish_yamnet':
                print(f"pip install torch-vggish-yamnet")
            elif pkg == 'python-multipart':
                print(f"pip install python-multipart")
            else:
                print(f"pip install {pkg}")
        return False
    
    return True

def check_reference_file():
    """检查参考特征文件是否存在"""
    csv_path = r"D:\Car\vehicile_voice_recoginition\audio_features_20250905_172636.csv"
    
    if not Path(csv_path).exists():
        print(f"✗ 参考特征文件不存在: {csv_path}")
        print("\n请确保以下文件存在:")
        print(f"  {csv_path}")
        print("\n或者修改 audio_api_server.py 中的 csv_path 变量为正确的路径")
        return False
    else:
        print(f"✓ 参考特征文件: {csv_path}")
        return True

def main():
    print("="*80)
    print("音频特征提取与相似度匹配API服务器启动检查")
    print("="*80)
    
    # 检查依赖
    print("\n1. 检查Python包依赖...")
    if not check_dependencies():
        sys.exit(1)
    
    # 检查参考文件
    print("\n2. 检查参考特征文件...")
    if not check_reference_file():
        sys.exit(1)
    
    print("\n✓ 所有检查通过！")
    print("="*80)
    
    # 启动服务器
    try:
        import uvicorn
        from audio_api_server import app
        
        print("正在启动服务器...")
        print("访问地址:")
        print("- Web界面: http://localhost:8000")
        print("- API文档: http://localhost:8000/docs")
        print("- 健康检查: http://localhost:8000/health")
        print("="*80)
        
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8000,
            reload=False,  # 生产环境建议设为False
            log_level="info"
        )
        
    except ImportError as e:
        print(f"导入模块失败: {e}")
        print("请确保 audio_api_server.py 文件在同一目录下")
        sys.exit(1)
    except Exception as e:
        print(f"启动服务器失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()