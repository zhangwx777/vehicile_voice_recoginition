import os
import torch
import librosa
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
import torchaudio
import torch.nn.functional as F
from datetime import datetime
import tempfile
import io
from sklearn.metrics.pairwise import cosine_similarity
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
from typing import List, Dict
import json

warnings.filterwarnings('ignore')

# 导入音频特征提取模型
try:
    from torch_vggish_yamnet import yamnet, vggish
except ImportError:
    print("警告: torch_vggish_yamnet 模块未安装，请安装相关依赖")
    print("pip install torch-vggish-yamnet")
    exit(1)

class AudioFeatureExtractor:
    """音频特征提取器"""
    
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
            
        except Exception as e:
            print(f"模型加载失败: {e}")
            raise
    
    def load_audio(self, audio_data):
        """加载音频数据"""
        try:
            # 使用librosa从字节数据加载音频
            audio, sr = librosa.load(io.BytesIO(audio_data), sr=self.target_sr, mono=True)
            
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
            print(f"加载音频失败: {str(e)}")
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
    
    def extract_features(self, audio_data):
        """提取音频特征"""
        # 加载音频
        x_in, in_sr = self.load_audio(audio_data)
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
                    print(f"YAMNet特征提取失败: {e}")
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
                    print(f"VGGish特征提取失败: {e}")
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
            print(f"特征提取过程失败: {str(e)}")
            return None, None

class SimilarityMatcher:
    """相似度匹配器"""
    
    def __init__(self, csv_path):
        """初始化相似度匹配器
        
        Args:
            csv_path: 包含特征数据的CSV文件路径
        """
        self.csv_path = csv_path
        self.reference_df = None
        self.yamnet_features = None
        self.vggish_features = None
        self.filenames = None
        self.load_reference_data()
    
    def load_reference_data(self):
        """加载参考特征数据"""
        try:
            # 检查文件是否存在
            if not Path(self.csv_path).exists():
                raise FileNotFoundError(f"参考特征文件不存在: {self.csv_path}")
            
            # 加载CSV文件
            self.reference_df = pd.read_csv(self.csv_path)
            print(f"成功加载参考数据: {self.reference_df.shape}")
            
            # 提取文件名
            self.filenames = self.reference_df['filename'].tolist()
            
            # 提取YAMNet特征
            yamnet_cols = [col for col in self.reference_df.columns if col.startswith('yamnet_feat_')]
            if yamnet_cols:
                self.yamnet_features = self.reference_df[yamnet_cols].values
                print(f"YAMNet特征维度: {self.yamnet_features.shape}")
            
            # 提取VGGish特征
            vggish_cols = [col for col in self.reference_df.columns if col.startswith('vggish_feat_')]
            if vggish_cols:
                self.vggish_features = self.reference_df[vggish_cols].values
                print(f"VGGish特征维度: {self.vggish_features.shape}")
                
        except Exception as e:
            print(f"加载参考数据失败: {e}")
            raise
    
    def find_most_similar(self, yamnet_feat, vggish_feat, top_k=2):
        """查找最相似的音频文件
        
        Args:
            yamnet_feat: YAMNet特征向量
            vggish_feat: VGGish特征向量
            top_k: 返回前k个最相似的结果
            
        Returns:
            List[Dict]: 包含文件名和相似度的结果列表
        """
        results = []
        
        try:
            # YAMNet相似度计算
            yamnet_similarities = None
            if yamnet_feat is not None and self.yamnet_features is not None:
                yamnet_feat_2d = yamnet_feat.reshape(1, -1)
                yamnet_similarities = cosine_similarity(yamnet_feat_2d, self.yamnet_features)[0]
            
            # VGGish相似度计算
            vggish_similarities = None
            if vggish_feat is not None and self.vggish_features is not None:
                vggish_feat_2d = vggish_feat.reshape(1, -1)
                vggish_similarities = cosine_similarity(vggish_feat_2d, self.vggish_features)[0]
            
            # 组合相似度（如果两个特征都存在，取平均值）
            combined_similarities = None
            if yamnet_similarities is not None and vggish_similarities is not None:
                combined_similarities = (yamnet_similarities + vggish_similarities) / 2.0
            elif yamnet_similarities is not None:
                combined_similarities = yamnet_similarities
            elif vggish_similarities is not None:
                combined_similarities = vggish_similarities
            else:
                return []
            
            # 获取前k个最相似的结果
            top_indices = np.argsort(combined_similarities)[-top_k:][::-1]
            
            for idx in top_indices:
                result = {
                    'filename': self.filenames[idx],
                    'similarity': float(combined_similarities[idx]),
                    'yamnet_similarity': float(yamnet_similarities[idx]) if yamnet_similarities is not None else None,
                    'vggish_similarity': float(vggish_similarities[idx]) if vggish_similarities is not None else None
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            print(f"相似度计算失败: {e}")
            return []

# 创建FastAPI应用
app = FastAPI(title="音频特征提取与相似度匹配API", version="1.0.0")

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局变量
extractor = None
matcher = None

@app.on_event("startup")
async def startup_event():
    """启动时初始化模型和匹配器"""
    global extractor, matcher
    
    try:
        # 初始化特征提取器
        print("初始化音频特征提取器...")
        extractor = AudioFeatureExtractor()
        
        # 初始化相似度匹配器
        csv_path = r"D:\Car\vehicile_voice_recoginition\audio_features_20250905_172636.csv"
        print(f"初始化相似度匹配器，参考文件: {csv_path}")
        matcher = SimilarityMatcher(csv_path)
        
        print("初始化完成！")
        
    except Exception as e:
        print(f"初始化失败: {e}")
        raise

@app.get("/", response_class=HTMLResponse)
async def get_html_page():
    """返回HTML页面"""
    return """
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>音频特征提取与相似度匹配</title>
        <style>
            * {
                box-sizing: border-box;
                margin: 0;
                padding: 0;
            }
            
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                line-height: 1.6;
                color: #333;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                display: flex;
                align-items: center;
                justify-content: center;
            }
            
            .container {
                background: white;
                border-radius: 20px;
                box-shadow: 0 20px 40px rgba(0,0,0,0.1);
                padding: 40px;
                max-width: 800px;
                width: 90%;
                backdrop-filter: blur(10px);
            }
            
            h1 {
                text-align: center;
                color: #4a5568;
                margin-bottom: 30px;
                font-size: 2.5em;
                font-weight: 600;
            }
            
            .upload-section {
                border: 3px dashed #cbd5e0;
                border-radius: 15px;
                padding: 40px;
                text-align: center;
                margin-bottom: 30px;
                background: #f8fafc;
                transition: all 0.3s ease;
            }
            
            .upload-section:hover {
                border-color: #667eea;
                background: #eef2ff;
            }
            
            .upload-section.dragover {
                border-color: #667eea;
                background: #eef2ff;
                transform: scale(1.02);
            }
            
            .file-input {
                display: none;
            }
            
            .upload-btn {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border: none;
                padding: 15px 30px;
                border-radius: 50px;
                cursor: pointer;
                font-size: 16px;
                font-weight: 600;
                transition: all 0.3s ease;
                box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
            }
            
            .upload-btn:hover {
                transform: translateY(-2px);
                box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
            }
            
            .file-info {
                margin-top: 15px;
                color: #4a5568;
                font-weight: 500;
            }
            
            .analyze-btn {
                background: linear-gradient(135deg, #48bb78 0%, #38a169 100%);
                color: white;
                border: none;
                padding: 15px 40px;
                border-radius: 50px;
                cursor: pointer;
                font-size: 16px;
                font-weight: 600;
                width: 100%;
                margin-bottom: 20px;
                transition: all 0.3s ease;
                box-shadow: 0 4px 15px rgba(72, 187, 120, 0.4);
            }
            
            .analyze-btn:hover:not(:disabled) {
                transform: translateY(-2px);
                box-shadow: 0 6px 20px rgba(72, 187, 120, 0.6);
            }
            
            .analyze-btn:disabled {
                background: #cbd5e0;
                cursor: not-allowed;
                box-shadow: none;
            }
            
            .loading {
                display: none;
                text-align: center;
                color: #667eea;
                font-weight: 600;
            }
            
            .spinner {
                border: 4px solid #e2e8f0;
                border-left: 4px solid #667eea;
                border-radius: 50%;
                width: 40px;
                height: 40px;
                animation: spin 1s linear infinite;
                margin: 0 auto 15px;
            }
            
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            
            .results {
                display: none;
                margin-top: 30px;
            }
            
            .result-item {
                background: #f8fafc;
                border: 2px solid #e2e8f0;
                border-radius: 15px;
                padding: 20px;
                margin-bottom: 15px;
                transition: all 0.3s ease;
            }
            
            .result-item:hover {
                border-color: #667eea;
                transform: translateY(-2px);
                box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            }
            
            .filename {
                font-size: 18px;
                font-weight: 600;
                color: #2d3748;
                margin-bottom: 10px;
            }
            
            .similarity {
                font-size: 16px;
                color: #4a5568;
                margin-bottom: 5px;
            }
            
            .similarity-bar {
                width: 100%;
                height: 8px;
                background: #e2e8f0;
                border-radius: 4px;
                overflow: hidden;
                margin-top: 8px;
            }
            
            .similarity-fill {
                height: 100%;
                background: linear-gradient(90deg, #48bb78, #38a169);
                transition: width 0.5s ease;
            }
            
            .error {
                color: #e53e3e;
                background: #fed7d7;
                border: 1px solid #feb2b2;
                border-radius: 10px;
                padding: 15px;
                margin-top: 20px;
                display: none;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎵 音频特征提取与相似度匹配</h1>
            
            <div class="upload-section" id="uploadSection">
                <div>
                    <p style="font-size: 18px; margin-bottom: 15px; color: #4a5568;">
                        点击选择音频文件或拖拽文件到此区域
                    </p>
                    <input type="file" id="audioFile" class="file-input" accept="audio/*">
                    <button class="upload-btn" onclick="document.getElementById('audioFile').click()">
                        📁 选择音频文件
                    </button>
                    <div id="fileInfo" class="file-info"></div>
                </div>
            </div>
            
            <button id="analyzeBtn" class="analyze-btn" disabled onclick="analyzeAudio()">
                🔍 分析音频并查找相似文件
            </button>
            
            <div id="loading" class="loading">
                <div class="spinner"></div>
                <p>正在分析音频特征，请稍候...</p>
            </div>
            
            <div id="results" class="results">
                <h2 style="color: #4a5568; margin-bottom: 20px;">📊 相似度匹配结果</h2>
                <div id="resultsList"></div>
            </div>
            
            <div id="error" class="error"></div>
        </div>
        
        <script>
            let selectedFile = null;
            
            // 文件选择处理
            document.getElementById('audioFile').addEventListener('change', function(e) {
                const file = e.target.files[0];
                handleFileSelection(file);
            });
            
            // 拖拽处理
            const uploadSection = document.getElementById('uploadSection');
            
            uploadSection.addEventListener('dragover', function(e) {
                e.preventDefault();
                uploadSection.classList.add('dragover');
            });
            
            uploadSection.addEventListener('dragleave', function(e) {
                e.preventDefault();
                uploadSection.classList.remove('dragover');
            });
            
            uploadSection.addEventListener('drop', function(e) {
                e.preventDefault();
                uploadSection.classList.remove('dragover');
                
                const files = e.dataTransfer.files;
                if (files.length > 0) {
                    const file = files[0];
                    if (file.type.startsWith('audio/')) {
                        handleFileSelection(file);
                    } else {
                        showError('请选择音频文件');
                    }
                }
            });
            
            function handleFileSelection(file) {
                if (file && file.type.startsWith('audio/')) {
                    selectedFile = file;
                    document.getElementById('fileInfo').innerHTML = `
                        <p><strong>已选择:</strong> ${file.name}</p>
                        <p><strong>大小:</strong> ${(file.size / 1024 / 1024).toFixed(2)} MB</p>
                        <p><strong>类型:</strong> ${file.type}</p>
                    `;
                    document.getElementById('analyzeBtn').disabled = false;
                    hideError();
                } else {
                    showError('请选择有效的音频文件');
                }
            }
            
            async function analyzeAudio() {
                if (!selectedFile) {
                    showError('请先选择音频文件');
                    return;
                }
                
                // 显示加载状态
                document.getElementById('loading').style.display = 'block';
                document.getElementById('results').style.display = 'none';
                document.getElementById('analyzeBtn').disabled = true;
                hideError();
                
                try {
                    const formData = new FormData();
                    formData.append('audio_file', selectedFile);
                    
                    const response = await fetch('/analyze_audio', {
                        method: 'POST',
                        body: formData
                    });
                    
                    if (!response.ok) {
                        const errorData = await response.json();
                        throw new Error(errorData.detail || '分析失败');
                    }
                    
                    const result = await response.json();
                    
                    // 隐藏加载状态
                    document.getElementById('loading').style.display = 'none';
                    document.getElementById('analyzeBtn').disabled = false;
                    
                    // 显示结果
                    displayResults(result.similar_files);
                    
                } catch (error) {
                    console.error('分析错误:', error);
                    document.getElementById('loading').style.display = 'none';
                    document.getElementById('analyzeBtn').disabled = false;
                    showError('分析失败: ' + error.message);
                }
            }
            
            function displayResults(similarFiles) {
                const resultsList = document.getElementById('resultsList');
                resultsList.innerHTML = '';
                
                if (!similarFiles || similarFiles.length === 0) {
                    resultsList.innerHTML = '<p style="text-align: center; color: #718096;">未找到相似的音频文件</p>';
                } else {
                    similarFiles.forEach((file, index) => {
                        const similarity = (file.similarity * 100).toFixed(1);
                        const item = document.createElement('div');
                        item.className = 'result-item';
                        item.innerHTML = `
                            <div class="filename">🎵 ${file.filename}</div>
                            <div class="similarity">
                                <strong>综合相似度:</strong> ${similarity}%
                            </div>
                            ${file.yamnet_similarity !== null ? 
                                `<div class="similarity">YAMNet相似度: ${(file.yamnet_similarity * 100).toFixed(1)}%</div>` : 
                                ''
                            }
                            ${file.vggish_similarity !== null ? 
                                `<div class="similarity">VGGish相似度: ${(file.vggish_similarity * 100).toFixed(1)}%</div>` : 
                                ''
                            }
                            <div class="similarity-bar">
                                <div class="similarity-fill" style="width: ${similarity}%"></div>
                            </div>
                        `;
                        resultsList.appendChild(item);
                    });
                }
                
                document.getElementById('results').style.display = 'block';
            }
            
            function showError(message) {
                const errorDiv = document.getElementById('error');
                errorDiv.textContent = message;
                errorDiv.style.display = 'block';
            }
            
            function hideError() {
                document.getElementById('error').style.display = 'none';
            }
        </script>
    </body>
    </html>
    """

@app.post("/analyze_audio")
async def analyze_audio(audio_file: UploadFile = File(...)):
    """分析上传的音频文件并返回相似度匹配结果"""
    global extractor, matcher
    
    if not extractor or not matcher:
        raise HTTPException(status_code=500, detail="服务未正确初始化")
    
    # 检查文件类型
    if not audio_file.content_type.startswith('audio/'):
        raise HTTPException(status_code=400, detail="请上传音频文件")
    
    try:
        # 读取上传的音频文件
        audio_data = await audio_file.read()
        
        # 提取特征
        yamnet_feat, vggish_feat = extractor.extract_features(audio_data)
        
        if yamnet_feat is None and vggish_feat is None:
            raise HTTPException(status_code=400, detail="音频特征提取失败，请检查音频文件格式")
        
        # 查找相似文件
        similar_files = matcher.find_most_similar(yamnet_feat, vggish_feat, top_k=2)
        
        return {
            "status": "success",
            "message": "音频分析完成",
            "file_info": {
                "filename": audio_file.filename,
                "content_type": audio_file.content_type,
                "size": len(audio_data)
            },
            "feature_info": {
                "yamnet_features": yamnet_feat.shape if yamnet_feat is not None else None,
                "vggish_features": vggish_feat.shape if vggish_feat is not None else None
            },
            "similar_files": similar_files
        }
        
    except Exception as e:
        print(f"处理音频文件时出错: {e}")
        raise HTTPException(status_code=500, detail=f"处理音频文件时出错: {str(e)}")

@app.get("/health")
async def health_check():
    """健康检查接口"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": extractor is not None and matcher is not None
    }

@app.get("/info")
async def get_system_info():
    """获取系统信息"""
    global matcher
    
    info = {
        "api_version": "1.0.0",
        "timestamp": datetime.now().isoformat(),
        "models_status": {
            "extractor_loaded": extractor is not None,
            "matcher_loaded": matcher is not None
        }
    }
    
    if matcher and matcher.reference_df is not None:
        info["reference_data"] = {
            "total_files": len(matcher.filenames),
            "yamnet_features_shape": matcher.yamnet_features.shape if matcher.yamnet_features is not None else None,
            "vggish_features_shape": matcher.vggish_features.shape if matcher.vggish_features is not None else None,
            "csv_path": matcher.csv_path
        }
    
    return info

if __name__ == "__main__":
    print("="*80)
    print("音频特征提取与相似度匹配API服务器")
    print("="*80)
    print("启动参数:")
    print("- 主机: 0.0.0.0")
    print("- 端口: 8000")
    print("- 重载: 是")
    print("="*80)
    print("访问地址:")
    print("- Web界面: http://localhost:8000")
    print("- API文档: http://localhost:8000/docs")
    print("- 健康检查: http://localhost:8000/health")
    print("- 系统信息: http://localhost:8000/info")
    print("="*80)
    
    uvicorn.run(
        "audio_api_server:app",  # 如果文件名不是audio_api_server.py，请修改这里
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )