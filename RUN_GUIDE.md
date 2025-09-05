# 🚀 车辆语音识别系统运行指南

## 📈 文档导航

- 📊 **虚拟音频特征分析**: [`AUDIO_FEATURES_ANALYSIS.md`](./AUDIO_FEATURES_ANALYSIS.md) - 详细的音频数据质量分析报告
- 🚀 **系统改进总结**: [`IMPROVEMENTS.md`](./IMPROVEMENTS.md) - 完整的系统优化历程
- 🔍 **推理指南**: [`INFERENCE_GUIDE.md`](./INFERENCE_GUIDE.md) - 详细的推理使用说明

## 📋 完整运行步骤总览

基于改进后的系统，以下是从环境准备到最终验证的完整运行步骤：

## 🎯 一键运行演示（推荐）

如果您想快速体验完整系统，可以运行一键演示：

```bash
# 安装依赖
pip install -r requirements.txt

# 运行完整演示（包含所有步骤）
python run_complete_demo.py
```

这个脚本会自动执行所有步骤，大约需要10-20分钟完成。

## 📖 分步运行指南

如果您想逐步了解每个环节，可以按以下步骤手动执行：

### 步骤1: 环境准备和依赖安装

```bash
# 1. 安装Python依赖
pip install -r requirements.txt

# 2. 验证环境配置
python setup_environment.py
```

**预期输出**: 
- ✅ 显示Python、PyTorch、CUDA版本信息
- ✅ 创建必要的目录结构
- ✅ 测试音频处理功能

---

### 步骤2: 生成或准备训练数据

```bash
# 快速生成优化版虚拟训练数据（推荐）
python generate_audio_data.py

# 或者生成复杂版本（计算量较大）
python generate_enhanced_data.py
```

**预期输出**:
- ✅ 生成5类车辆音频数据（sedan, suv, truck, motorcycle, bus）
- ✅ 每类100个样本，总计500个音频文件
- ✅ 保存数据统计信息

**📊 虚拟音频特征分析结果**:

为了验证生成的音频数据质量，系统会自动分析各车辆类型的特征差异：

| 车辆类型 | 主频率 (Hz) | 能量水平 | 频谱质心 (Hz) | 特征描述 |
|----------|-------------|----------|---------------|----------|
| 🏎️ **Motorcycle** | 228.8 ± 1.7 | 0.128 | 1225.8 | **高频、尖锐、激进** |
| 🚗 **Sedan** | 115.0 ± 1.6 | 0.171 | 1012.0 | **中频、平稳、温和** |
| 🚙 **SUV** | 84.3 ± 15.5 | 0.141 | 994.1 | **中低频、厚重、强劲** |
| 🚌 **Bus** | 63.9 ± 12.2 | 0.138 | 1240.4 | **低频、宽厚、稳重** |
| 🚛 **Truck** | 30.0 ± 0.0 | 0.097 | 1061.8 | **超低频、深沉、重型** |

**✅ 特征区分度分析**:
- **主频率变化范围**: 198.8 Hz（从卡车30Hz到摩托车229Hz）
- **能量变化范围**: 0.074（显著差异）
- **频谱质心变化范围**: 246.3 Hz（良好区分）

**🔍 数据质量验证**:
```bash
# 可选：分析生成音频的特征差异
python audio_feature_analyzer.py
```

**替代方案**: 如果您有真实音频数据，请将其按以下结构放置：
```
vehicle_audio_data/
├── sedan/
│   ├── file1.wav
│   └── file2.wav
├── suv/
│   ├── file1.wav
│   └── file2.wav
└── ...
```

---

### 步骤3: 配置系统参数

```bash
# 验证和优化系统配置
python config_manager.py
```

**预期输出**:
- ✅ 验证所有配置参数的有效性
- ✅ 显示详细的系统配置摘要
- ✅ 根据硬件给出优化建议
- ✅ 保存配置文件用于训练

---

### 步骤4: 执行模型训练

**方式1: 基础训练**
```bash
python train.py
```

**方式2: 带监控的训练（推荐）**
```bash
python train_monitor.py
```

**预期输出**:
- ✅ 显示数据集统计信息（训练/验证/测试集大小）
- ✅ 实时显示训练进度和性能指标
- ✅ 自动早停机制防止过拟合
- ✅ 保存最佳模型和训练历史可视化

**训练参数**（可在settings.py中调整）:
- 训练轮次: 50 epochs
- 批次大小: 16
- 学习率: 0.001
- 早停耐心值: 10轮

**预计时间**: 
- CPU: 15-30分钟
- GPU: 3-10分钟

---

### 步骤5: 模型评估和验证

```bash
# 全面评估训练好的模型
python model_evaluation.py
```

**预期输出**:
- ✅ 测试集性能评估（准确率、F1分数）
- ✅ 混淆矩阵可视化
- ✅ 个别样本预测分析
- ✅ 推理速度基准测试
- ✅ 详细的评估报告（JSON + 文本格式）

**生成的文件**:
- `results/evaluation/test_confusion_matrix.png` - 混淆矩阵
- `results/evaluation/classification_report.txt` - 分类报告
- `results/evaluation/individual_sample_results.csv` - 样本结果
- `results/evaluation/evaluation_summary.json` - 评估摘要

---

### 步骤6: 推理测试和可视化

**重要提示**: 运行推理模块时，请确保使用完整的文件名 `vehicle_inference.py`，**不要省略 `.py` 扩展名**，否则会出现 "文件找不到" 错误。

**单文件推理**:
```bash
# 预测单个音频文件（默认带可视化）
python vehicle_inference.py vehicle_audio_data\motorcycle\motorcycle_001.wav

# 预测单个音频文件（禁用可视化）
python vehicle_inference.py vehicle_audio_data\bus\bus_050.wav --no-visualize
```

**批量推理**:
```bash
# 批量处理音频目录（默认带可视化）
python vehicle_inference.py vehicle_audio_data\suv --batch --output results\suv_results.csv

# 处理所有测试数据（禁用可视化以提高速度）
python vehicle_inference.py vehicle_audio_data --batch --output results\all_vehicles_results.csv --no-visualize
```

**交互式模式**:
```bash
# 进入交互式预测模式
python vehicle_inference.py --interactive
```

**推理模块命令参数表**:

| 参数 | 说明 | 示例 |
|------|------|------|
| `audio_path` | 音频文件或目录路径 | `vehicle_audio_data\motorcycle\motorcycle_001.wav` |
| `--batch` | 启用批量处理模式 | `--batch` |
| `--interactive`, `-i` | 启用交互式模式 | `-i` |
| `--visualize`, `-v` | 生成可视化结果（默认已启用） | `-v` |
| `--no-visualize` | 禁用可视化结果生成 | `--no-visualize` |
| `--output`, `-o` | 批量处理结果输出文件 | `-o results\batch_output.csv` |
| `--visualize-top` | 可视化最高置信度的结果数量 | `--visualize-top 5` |

**预期输出**:
- ✅ 预测结果（车辆类型 + 置信度）
- ✅ 推理时间统计
- ✅ 可视化图表（音频特征 + 预测分布）
- ✅ 批量处理统计报告

---

## 🔍 运行结果验证

完成所有步骤后，您应该看到以下文件结构：

```
vehicle_voice_recognition/
├── vehicle_audio_data/          # 训练数据
├── models/                      # 保存的模型
│   ├── best_model.pth          # 最佳模型权重
│   └── label_mapping.json      # 类别映射
├── results/                     # 结果和可视化
│   ├── confusion_matrix.png    # 混淆矩阵
│   ├── training_history.png    # 训练历史
│   ├── evaluation/             # 评估结果
│   ├── audio_visualizations/   # 音频特征图
│   └── prediction_visualizations/ # 预测结果图
├── logs/                       # 日志文件
└── checkpoints/               # 训练检查点
```

## 🎯 性能期望

在虚拟数据上的典型性能指标：

- **训练准确率**: 95%+ 
- **验证准确率**: 90%+
- **测试准确率**: 85%+
- **推理时间**: <100ms per file
- **模型大小**: ~2-5MB

## 🔧 故障排除

### 常见问题

1. **找不到文件 `vehicle_inference`**:
   ```bash
   # 错误原因: 命令中遗漏了 .py 扩展名
   # 错误命令: python vehicle_inference vehicle_audio_data\motorcycle\motorcycle_001.wav
   # 正确命令: python vehicle_inference.py vehicle_audio_data\motorcycle\motorcycle_001.wav
   ```

2. **CUDA内存不足**:
   ```bash
   # 减少批次大小
   # 在 settings.py 中修改 TRAINING_CONFIG['batch_size'] = 8
   ```

3. **音频文件格式错误**:
   ```bash
   # 确保音频文件为支持的格式：.wav, .mp3, .flac, .ogg, .m4a
   ```

4. **模型文件不存在**:
   ```bash
   # 确保完成了训练步骤
   python train.py
   ```

5. **依赖库缺失**:
   ```bash
   pip install -r requirements.txt
   ```

### 日志查看

所有运行日志都保存在 `logs/` 目录中，包含详细的错误信息和调试数据。

## 🚀 下一步建议

1. **使用真实数据**: 将虚拟数据替换为真实的车辆音频数据
2. **超参数调优**: 在 `settings.py` 中调整训练参数
3. **模型改进**: 尝试不同的网络架构或特征提取方法
4. **部署优化**: 使用模型量化和优化技术提升推理速度

## 📞 技术支持

如果在运行过程中遇到问题，请：

1. 查看日志文件中的详细错误信息
2. 确认所有依赖已正确安装
3. 验证音频文件格式和路径正确性
4. 检查系统内存和存储空间是否充足

---

**🎉 祝您使用愉快！这个改进后的车载语音识别系统已经具备了生产级别的功能和性能。**