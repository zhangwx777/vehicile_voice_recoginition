# 车辆语音识别系统

## 项目简介

这是一个基于深度学习的车辆语音识别系统，能够通过音频信号识别不同的车辆个体。系统使用卷积神经网络(CNN)对车辆音频特征进行分析，实现高精度的车辆个体识别。

## 功能特性

- **车辆个体识别**: 支持识别25个不同的车辆个体（包括轿车、SUV、卡车、摩托车、公交车）
- **音频特征提取**: 使用梅尔频谱图(Mel-spectrogram)提取音频特征
- **深度学习模型**: 基于CNN的车辆音频分类模型
- **可视化功能**: 提供音频特征和预测结果的可视化
- **实时推理**: 支持单个音频文件的快速推理
- **模型评估**: 完整的模型性能评估和指标分析

## 系统架构

```
vehicle_voice_recognition/
├── core/                    # 核心模块
│   ├── config_manager.py   # 配置管理
│   ├── logger.py           # 日志系统
│   ├── model.py            # CNN模型定义
│   └── settings.py         # 系统配置
├── data/                    # 数据处理模块
│   ├── preprocessor.py     # 音频预处理
│   ├── loader.py           # 数据加载器
│   ├── individual_loader.py # 个体数据加载
│   ├── label_manager.py    # 标签管理
│   └── augmentation.py     # 数据增强
├── training/                # 训练模块
│   ├── trainer.py          # 模型训练器
│   ├── early_stopping.py   # 早停机制
│   └── individual_monitor.py # 训练监控
├── evaluation/              # 评估模块
│   ├── evaluator.py        # 模型评估器
│   └── model_evaluation.py # 评估脚本
├── scripts/                 # 脚本模块
│   ├── individual_inference.py # 推理脚本
│   └── recognizer.py       # 识别器
├── utils/                   # 工具模块
│   ├── visualization.py    # 可视化工具
│   ├── individual_visualization.py # 个体可视化
│   └── helpers.py          # 辅助函数
├── models/                  # 模型文件
├── results/                 # 结果输出
├── logs/                    # 日志文件
└── vehicle_audio_data/      # 音频数据
```

## 环境要求

### Python版本
- Python 3.8+

### 依赖包
```
torch>=1.9.0
torchaudio>=0.9.0
librosa>=0.8.1
numpy>=1.21.0
matplotlib>=3.3.4
seaborn>=0.11.1
scikit-learn>=0.24.2
soundfile>=0.10.3
psutil>=5.8.0
```

## 安装说明

1. **克隆项目**
```bash
git clone <repository-url>
cd vehicle_voice_recognition
```

2. **创建虚拟环境**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate     # Windows
```

3. **安装依赖**
```bash
pip install torch torchaudio librosa numpy matplotlib seaborn scikit-learn soundfile psutil
```

4. **验证安装**
```bash
python -c "from core import settings; print('安装成功')"
```

## 使用说明

### 单个音频文件推理

```bash
# 基本推理（包含可视化）
python -m scripts.individual_inference --audio "path/to/audio.wav"

# 不生成可视化
python -m scripts.individual_inference --audio "path/to/audio.wav" --no-visualize

# 使用自定义模型
python -m scripts.individual_inference --audio "path/to/audio.wav" --model "path/to/model.pth"
```

### 命令行参数

- `--audio`: 音频文件路径（必需）
- `--model`: 模型文件路径（可选，默认: models/individual_best_model.pth）
- `--mapping`: 标签映射文件路径（可选，默认: vehicle_audio_data/individual_recognition/individual_label_mapping.json）
- `--visualize`: 生成可视化结果（默认启用）
- `--no-visualize`: 禁用可视化

### 模型训练

```bash
# 训练个体识别模型
python -m training.individual_monitor
```

### 模型评估

```bash
# 评估模型性能
python -m evaluation.model_evaluation
```

## 输出说明

### 推理结果
系统会输出以下信息：
- 预测的车辆个体ID
- 车辆类型（sedan、suv、truck、motorcycle、bus）
- 个体编号
- 置信度分数
- 推理时间

### 可视化文件
- **音频特征可视化**: `results/audio_visualizations/`
- **预测结果可视化**: `results/prediction_visualizations/`

## 支持的车辆类型

| 车辆类型 | 个体数量 | 编号范围 |
|---------|---------|----------|
| 轿车 (sedan) | 5 | 000-004 |
| SUV | 5 | 000-004 |
| 卡车 (truck) | 5 | 000-004 |
| 摩托车 (motorcycle) | 5 | 000-004 |
| 公交车 (bus) | 5 | 000-004 |

总计：25个不同的车辆个体

## 音频格式要求

- **采样率**: 16kHz
- **格式**: WAV, MP3, FLAC等（librosa支持的格式）
- **时长**: 建议3秒以上
- **质量**: 清晰的车辆音频，噪音较少

## 性能指标

当前模型在测试集上的性能：
- **准确率**: 根据最新训练结果而定
- **推理时间**: 约1.8秒/音频文件
- **模型大小**: 约375KB

## 故障排除

### 常见问题

1. **导入错误**
   ```
   解决方案：确保在项目根目录运行命令，使用 python -m 方式运行脚本
   ```

2. **音频文件无法读取**
   ```
   解决方案：检查音频文件路径和格式，确保文件存在且格式正确
   ```

3. **模型文件不存在**
   ```
   解决方案：确保 models/individual_best_model.pth 文件存在，或重新训练模型
   ```

4. **CUDA相关错误**
   ```
   解决方案：系统会自动使用CPU，无需GPU也可正常运行
   ```

### 日志文件
系统运行日志保存在 `logs/` 目录下，文件名格式为 `vehicle_voice_recognition_YYYYMMDD_HHMMSS.log`

## 开发说明

### 代码结构
- 遵循模块化设计原则
- 使用配置文件管理系统参数
- 完整的日志记录系统
- 异常处理和错误恢复机制

### 扩展功能
- 支持添加新的车辆类型
- 可自定义音频预处理参数
- 支持不同的深度学习模型架构

## 许可证

本项目仅供学习和研究使用。

## 更新日志

### v1.0.0
- 初始版本发布
- 支持25个车辆个体识别
- 完整的训练和推理流程
- 可视化功能
- 模型评估工具

---

如有问题或建议，请查看日志文件或联系开发团队。