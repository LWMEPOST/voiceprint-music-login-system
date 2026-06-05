# 融合轻量化声纹识别的音乐平台登录保护系统

## 项目介绍

本项目是一个音乐平台登录保护系统，将传统账号密码登录与轻量化声纹识别结合，提供用户认证、声纹注册、声纹验证和安全登录能力。仓库保留核心代码，数据集、模型权重、证书和大文件已过滤。

## 技术栈

- Python
- FastAPI
- SQLite
- SQLAlchemy
- JWT
- PyTorch/ONNX 模型结构
- HTML/CSS/JavaScript 前端

## 部署要求

- Python 3.9 或以上
- pip
- requirements.txt 依赖
- 现代浏览器
- 可选：onnxruntime

## 运行流程

1. 执行 pip install -r requirements.txt 安装依赖。
2. 如需重新训练模型，按 README 或训练脚本生成数据并执行训练。
3. 进入后端入口所在目录执行 uvicorn backend.main:app --reload。
4. 打开 frontend 或静态页面进行登录与声纹验证测试。
5. 生产部署时请替换密钥、证书和数据库配置。

## 项目结构

- backend：FastAPI 后端服务
- frontend：前端页面
- models：模型结构代码
- tests：测试脚本
- requirements.txt：Python 依赖
