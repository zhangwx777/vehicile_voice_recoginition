# 车辆语音识别系统

基于深度学习和向量相似度的车辆语音识别系统，能够通过音频特征识别不同类型的车辆。

## 项目特点

- 🎯 **高精度识别**：基于向量相似度的识别算法，准确率达95%以上
- 🚀 **快速响应**：平均识别时间1-2秒
- 📊 **可视化分析**：自动生成音频特征图表和识别结果
- 🔧 **易于使用**：简单的命令行接口和Python API
- 📈 **可扩展**：支持添加新的车辆档案

## 支持的车辆类型

- 轿车 (Sedan)
- SUV
- 卡车 (Truck)
- 公交车 (Bus)
- 摩托车 (Motorcycle)

## 系统要求

- Python 3.8+
- PyTorch 1.9+
- librosa
- matplotlib
- seaborn
- numpy
- scikit-learn

## 安装说明

1. **克隆项目**
```bash
git clone <repository-url>
cd vehicle_voice_recognition
```

2. **安装依赖**
```bash
pip install torch torchvision torchaudio
pip install librosa matplotlib seaborn numpy scikit-learn
```

3. **验证安装**
```bash
python local_recognition.py --help
```

## 快速开始

### 基本识别

```bash
# 识别单个音频文件
python local_recognition.py "path/to/audio.wav"

# 使用示例音频
python local_recognition.py "vehicle_audio_data/individual_recognition/individual_vehicles/sedan_001/sedan_001_sample_000.wav"
```

### 可视化分析

```bash
# 生成可视化结果
python visualization/local_visualizer.py "path/to/audio.wav"

# 结果将保存在 results/visualizations/ 目录
```

## 项目结构

```
vehicle_voice_recognition/
├── README.md                    # 项目说明文档
├── config.py                    # 配置文件
├── local_recognition.py         # 本地识别脚本
├── train_and_inference.py       # 训练和推理脚本
├── enhanced_feature_extractor.py # 增强特征提取器
├── vector_similarity_engine.py  # 向量相似度引擎
│
├── core/                        # 核心模块
│   ├── __init__.py
│   ├── logger.py               # 日志模块
│   └── unified_engine.py       # 统一识别引擎
│
├── data/                        # 数据处理模块
│   ├── __init__.py
│   └── preprocessor.py         # 音频预处理器
│
├── visualization/               # 可视化模块
│   ├── __init__.py
│   ├── README.md
│   ├── enhanced_visualizer.py  # 增强可视化器
│   └── local_visualizer.py     # 本地可视化器
│
├── models/                      # 模型和数据库
│   ├── similarity_database.json # 相似度数据库
│   ├── individual_best_model.pth # 训练好的模型
│   ├── individual_label_mapping.json # 标签映射
│   ├── training_history.json   # 训练历史
│   └── evaluation_results.json # 评估结果
│
├── vehicle_audio_data/          # 音频数据
│   └── individual_recognition/
│       ├── individual_vehicles/ # 各车辆音频样本
│       ├── individual_label_mapping.json
│       └── vehicle_registry.json
│
├── results/                     # 结果输出
│   └── visualizations/         # 可视化结果
│
└── logs/                        # 日志文件
```

## 使用说明

### 1. 音频识别（默认带可视化）

**基本用法（自动生成可视化）：**
```bash
python local_recognition.py "audio_file.wav"
```

**输出示例：**
```
识别文件: sedan_001_sample_000.wav (模式: similarity)
----------------------------------------

📊 可视化结果已生成:
  音频特征图: results\audio_features_sedan_001_sample_000_20250910_191512.png
  识别结果图: results\recognition_result_sedan_001_20250910_191513.png
  详细数据: results\recognition_data_sedan_001_sample_000_20250910_191514.json

✅ 识别成功!
车辆ID: sedan_001
置信度: 0.9863
识别方法: similarity
处理时间: 1.979秒

相似度排名:
  1. 车辆sedan_001: 0.9863
  2. 车辆truck_003: 0.9695
  3. 车辆suv_003: 0.9622
```

**可视化选项：**
```bash
# 禁用可视化（仅文本输出）
python local_recognition.py "audio_file.wav" --no-viz

# 仅生成可视化（无文本输出）
python local_recognition.py "audio_file.wav" --viz-only

# 设置置信度阈值
python local_recognition.py "audio_file.wav" --threshold 0.8
```

### 2. 独立可视化分析

**单独生成可视化：**
```bash
python visualization/local_visualizer.py "audio_file.wav"
```

**输出文件：**
- `audio_features_*.png` - 音频特征分析图
- `recognition_result_*.png` - 识别结果图表
- `recognition_data_*.json` - 详细数据

### 3. 添加新车辆档案

```python
from core.unified_engine import UnifiedVehicleRecognitionEngine

engine = UnifiedVehicleRecognitionEngine()
engine.add_vehicle_profile(
    vehicle_id="new_vehicle_001",
    audio_path="path/to/new_vehicle_audio.wav",
    metadata={"type": "sedan", "model": "Toyota Camry"}
)
engine.save_database()
```

### 4. 训练新模型

```bash
# 运行完整训练流程
python train_and_inference.py
```

## API 参考

### UnifiedVehicleRecognitionEngine

主要的识别引擎类。

```python
from core.unified_engine import UnifiedVehicleRecognitionEngine

# 初始化引擎
engine = UnifiedVehicleRecognitionEngine()

# 识别音频
result, audio_info = engine.recognize("audio_file.wav")

# 获取统计信息
stats = engine.get_stats()
```

### 主要方法

- `recognize(audio_path)` - 识别音频文件
- `add_vehicle_profile(vehicle_id, audio_path, metadata)` - 添加车辆档案
- `save_database(path)` - 保存数据库
- `load_database(path)` - 加载数据库
- `get_stats()` - 获取统计信息

## 配置说明

主要配置在 `config.py` 中：

```python
# 路径配置
PATH_CONFIG = {
    'models_dir': 'models',
    'data_dir': 'vehicle_audio_data',
    'results_dir': 'results'
}

# 音频配置
AUDIO_CONFIG = {
    'sample_rate': 22050,
    'duration': 5.0,
    'n_mels': 128
}

# 相似度配置
SIMILARITY_CONFIG = {
    'threshold': 0.7,
    'top_k': 5
}
```

## 性能指标

- **识别准确率**: 95%+
- **平均处理时间**: 1-2秒
- **支持音频格式**: WAV, MP3, FLAC, OGG, M4A
- **数据库容量**: 当前25个车辆档案

## 故障排除

### 常见问题

1. **音频文件无法识别**
   - 检查文件路径是否正确
   - 确认音频格式支持
   - 检查文件是否损坏

2. **识别准确率低**
   - 确保音频质量良好
   - 检查背景噪音
   - 考虑重新训练模型

3. **可视化生成失败**
   - 检查matplotlib安装
   - 确认输出目录权限
   - 查看日志文件

### 日志查看

```bash
# 查看最新日志
tail -f logs/system.log
```

## 开发说明

### 添加新功能

1. 在相应模块中添加代码
2. 更新配置文件
3. 添加测试用例
4. 更新文档

### 代码规范

- 使用Python 3.8+语法
- 遵循PEP 8代码风格
- 添加类型注解
- 编写文档字符串

## 许可证

本项目采用 MIT 许可证。

## 贡献

欢迎提交 Issue 和 Pull Request！

## 更新日志

### v1.0.0 (2025-01-10)
- 初始版本发布
- 支持基于相似度的车辆识别
- 集成可视化功能
- 完整的命令行接口

---

如有问题，请查看文档或提交 Issue。