# 车辆声纹识别工具

## 快速使用

识别音频文件并生成可视化结果：

```bash
python visualization/local_visualizer.py "音频文件路径"
```

## 示例

```bash
# 识别轿车音频
python visualization/local_visualizer.py "vehicle_audio_data/individual_recognition/individual_vehicles/sedan_001/sedan_001_sample_000.wav"
```

## 输出结果

- 音频特征图片（波形、频谱、MFCC）
- 识别结果可视化图片
- 详细结果JSON文件

所有文件保存在 `results/visualizations/` 目录下