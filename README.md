# 🚗 车辆语音识别系统

基于深度学习的车辆语音识别系统，使用ECAPA-TDNN模型进行特征提取和相似度匹配。

## ✨ 主要特性

- **高精度识别**: 基于ECAPA-TDNN深度学习模型
- **智能缓存**: 多层缓存机制，提升处理效率
- **多线程并发**: 支持批量处理和并发操作
- **可视化分析**: 提供音频波形、频谱图和识别结果的可视化
- **统一异常处理**: 完善的错误处理和恢复机制

## 🚀 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 基本使用

```python
# 单个音频识别
from local_recognition import recognize_audio
result = recognize_audio("path/to/audio.wav")
print(f"识别结果: {result}")

# 批量处理
from optimized_feature_extractor import OptimizedFeatureExtractor
extractor = OptimizedFeatureExtractor()
features = extractor.extract_features_batch(audio_files)
```

### 命令行使用

```bash
# 添加音频样本
python database_management_cli.py add-sample path/to/audio.wav vehicle_type

# 识别音频
python local_recognition.py path/to/test_audio.wav

# 可视化分析
python audio_visualizer.py path/to/audio.wav
```

## 📊 性能指标

- **处理速度**: ~50ms/样本 (GPU模式)
- **缓存命中率**: 85%+
- **识别准确率**: 90%+ (测试数据集)
- **并发支持**: 多线程处理

## ⚙️ 配置选项

主要配置参数在 `config.py` 中：

```python
# 音频处理
AUDIO_CONFIG = {
    'sample_rate': 16000,
    'duration': 3.0
}

# 相似度阈值
SIMILARITY_CONFIG = {
    'threshold': 0.7
}

# 缓存设置
CACHE_CONFIG = {
    'enable_cache': True,
    'max_cache_size': 1000
}
```

## 📁 项目结构

```
├── core/                    # 核心模块
│   ├── common_utils.py     # 通用工具
│   ├── exceptions.py       # 异常处理
│   └── logger.py          # 日志系统
├── optimized_feature_extractor.py  # 特征提取器
├── advanced_similarity_engine.py   # 相似度引擎
├── local_recognition.py    # 本地识别接口
├── audio_visualizer.py     # 音频可视化
└── config.py              # 配置文件
```

## 🐛 故障排除

### 常见问题

1. **模型加载失败**: 检查预训练模型文件是否存在
2. **音频处理错误**: 验证音频文件格式和完整性
3. **内存不足**: 减少批处理大小，启用缓存清理
4. **识别准确率低**: 调整相似度阈值，增加训练样本

