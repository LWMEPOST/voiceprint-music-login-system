# 融合轻量化声纹识别的音乐平台登录保护系统

本项目将原本基于 Streamlit 单体架构的传统声纹识别应用，重构为具备**前后端分离**、采用**从零训练的轻量化深度学习网络**的工业级应用。系统旨在为音乐平台提供一套快速、安全、抗噪的登录保护机制，同时兼容传统密码登录。

## 项目结构
```
D:\XM\LYX\
├── backend/            # FastAPI 后端服务代码
│   ├── auth.py         # JWT与密码认证
│   ├── database.py     # SQLite 数据库配置
│   ├── main.py         # 核心 API 接口
│   ├── models.py       # SQLAlchemy 模型
│   ├── voiceprint.py   # 声纹识别引擎 (ONNX 推理)
├── dataset/            # 数据集生成与预处理
│   ├── generate_data.py # 合成虚拟声纹数据
│   ├── preprocess.py    # VAD、降噪、Mel特征提取
├── docs/               # 文档 (设计说明书、测试报告)
├── frontend/           # 移动端 H5 界面 (HTML/JS/CSS)
├── models/             # 模型定义与权重
│   ├── network.py      # 轻量化 CNN 模型结构
│   ├── weights/        # 训练好的模型 (.pth, .onnx)
├── tests/              # 自动化测试脚本
│   ├── eval_system.py  # 精度、速度、抗噪性评估
├── train.py            # 模型训练脚本
├── requirements.txt    # 依赖清单
```

## 环境要求
- Python 3.13 (或 3.9+)
- Windows / Linux / macOS
- 浏览器 (Chrome/Edge/Safari 等支持 Web Audio API)

## 快速启动指南

### 1. 安装依赖
```bash
python -m venv .venv
# 激活虚拟环境 (Windows)
.\.venv\Scripts\activate
# (Linux/macOS) source .venv/bin/activate

pip install -r requirements.txt
pip install onnxruntime
```

### 2. 生成数据与模型训练 (如需重新训练)
本项目自带生成脚本，无需下载庞大的公开数据集：
```bash
# 1. 生成虚拟声纹数据集 (10 个说话人)
python dataset/generate_data.py

# 2. 数据预处理 (特征提取)
python dataset/preprocess.py

# 3. 训练轻量化模型并导出 ONNX
python train.py
```

### 3. 运行自动化测试
```bash
# 执行端到端性能评估
$env:PYTHONPATH="."  # (Windows PowerShell)
python tests/eval_system.py
```

### 4. 启动服务
```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```
启动后，在浏览器访问 `http://localhost:8000` 即可体验适配手机端的声纹注册与登录。
*注意：如需在手机上访问，需保证手机与电脑在同一局域网，由于浏览器安全限制，麦克风访问通常要求 `localhost` 或 `https`。*
