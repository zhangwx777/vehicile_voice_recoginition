# 音频特征提取与相似度匹配API

这是一个基于深度学习的音频特征提取和相似度匹配系统，使用YAMNet和VGGish模型提取音频特征，并与预存的特征库进行相似度对比。

## 功能特点

* 🎵 支持多种音频格式（WAV、MP3、FLAC、M4A、AAC、OGG、WMA）
* 🔍 使用YAMNet和VGGish双模型特征提取
* 📊 余弦相似度计算，返回最相似的前2个文件
* 🌐 Web界面，支持拖拽上传
* 🚀 FastAPI后端，自动生成API文档
* 📱 响应式设计，支持移动端

## 安装依赖

### 方法1：使用requirements.txt（推荐）

```bash
pip install -r requirements.txt
```

### 方法2：手动安装主要依赖

```bash
pip install torch torchaudio librosa pandas numpy scikit-learn fastapi uvicorn python-multipart torch-vggish-yamnet tqdm
```

## 配置设置

1. **修改参考特征文件路径** ：
   在 `audio_api_server.py` 文件中找到以下行：

```python
   csv_path = r"D:\Car\vehicile_voice_recoginition\audio_features_20250905_172636.csv"
```

   将路径修改为您的实际特征文件路径。

1. **确保特征文件格式正确** ：
   CSV文件应包含以下列：

* `filename`: 音频文件名
* `yamnet_feat_000` 到 `yamnet_feat_xxx`: YAMNet特征
* `vggish_feat_000` 到 `vggish_feat_xxx`: VGGish特征

## 启动服务

### 方法1：使用启动脚本（推荐）

```bash
python run_server.py
```

### 方法2：直接启动

```bash
python audio_api_server.py
```

### 方法3：使用uvicorn

```bash
uvicorn audio_api_server:app --host 0.0.0.0 --port 8000 --reload
```

## 访问服务

启动成功后，您可以通过以下地址访问：

* **Web界面** : http://localhost:8000
* **API文档** : http://localhost:8000/docs
* **健康检查** : http://localhost:8000/health
* **系统信息** : http://localhost:8000/info

## API接口

### POST /analyze_audio

分析上传的音频文件并返回相似度匹配结果。

**请求参数：**

* `audio_file`: 音频文件（multipart/form-data）

**响应示例：**

```json
{
  "status": "success",
  "message": "音频分析完成",
  "file_info": {
    "filename": "test.wav",
    "content_type": "audio/wav",
    "size": 1024000
  },
  "feature_info": {
    "yamnet_features": [1024],
    "vggish_features": [128]
  },
  "similar_files": [
    {
      "filename": "similar1.wav",
      "similarity": 0.95,
      "yamnet_similarity": 0.94,
      "vggish_similarity": 0.96
    },
    {
      "filename": "similar2.wav", 
      "similarity": 0.89,
      "yamnet_similarity": 0.88,
      "vggish_similarity": 0.90
    }
  ]
}
```

## 使用Web界面

1. 打开浏览器访问 http://localhost:8000
2. 点击选择文件或直接拖拽音频文件到上传区域
3. 点击"分析音频并查找相似文件"按钮
4. 等待处理完成，查看相似度匹配结果

## 故障排除

### 常见问题

1. **模块导入错误**

   ```
   ImportError: torch_vggish_yamnet
   ```

   **解决方案** : 安装缺失的包

   ```bash
   pip install torch-vggish-yamnet
   ```
2. **参考文件不存在**

   ```
   FileNotFoundError: 参考特征文件不存在
   ```

   **解决方案** : 检查并修改 `audio_api_server.py` 中的文件路径
3. **音频加载失败**

   ```
   音频特征提取失败，请检查音频文件格式
   ```

   **解决方案** :

   * 确保上传的是有效的音频文件
   * 检查音频文件不是损坏的
   * 尝试转换为WAV格式
4. **内存不足**
   如果处理大文件时遇到内存问题，可以：

   * 减小音频文件大小
   * 增加系统内存
   * 调整batch处理参数

### 性能优化

1. **使用GPU加速** （如果可用）：
   模型会自动检测并使用GPU
2. **调整音频参数** ：
   可在 `AudioFeatureExtractor` 类中调整 `target_sr` 等参数
3. **批处理优化** ：
   对于大量文件，可以修改代码支持批处理

## 技术架构

* **后端框架** : FastAPI
* **深度学习** : PyTorch
* **音频处理** : Librosa, TorchAudio
* **特征提取** : YAMNet, VGGish
* **相似度计算** : 余弦相似度（scikit-learn）
* **前端** : HTML5 + JavaScript + CSS3

## 系统要求

* Python 3.7+
* 2GB+ RAM（推荐4GB+）
* 支持的操作系统：Windows、Linux、macOS

## 许可证

本项目仅用于学习和研究目的。请遵守相关模型的使用许可证。
