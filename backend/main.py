import os
import json
import shutil
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Form, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from datetime import timedelta

from .database import engine, Base, get_db
from . import models, auth
from .voiceprint import engine as vp_engine

Base.metadata.create_all(bind=engine)

app = FastAPI(title="Voiceprint Auth API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "backend/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/api/register")
async def register(
    username: str = Form(...),
    password: str = Form(...),
    audio: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    # 检查用户是否存在
    db_user = db.query(models.User).filter(models.User.username == username).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Username already registered")
        
    # 保存音频
    audio_path = os.path.join(UPLOAD_DIR, f"{username}_reg.webm")
    with open(audio_path, "wb") as buffer:
        shutil.copyfileobj(audio.file, buffer)
        
    # 提取声纹
    try:
        embedding = vp_engine.extract_feature(audio_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Audio processing failed: {str(e)}")
        
    # 保存到数据库
    hashed_pwd = auth.get_password_hash(password)
    new_user = models.User(
        username=username,
        hashed_password=hashed_pwd,
        # embedding 现在是一个包含 "cnn_embedding" 和 "mfcc_sequence" 的字典
        voiceprint_embedding=json.dumps(embedding)
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    
    return {"message": "User registered successfully"}

@app.post("/api/login/password")
async def login_password(
    username: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db)
):
    db_user = db.query(models.User).filter(models.User.username == username).first()
    if not db_user or not auth.verify_password(password, db_user.hashed_password):
        raise HTTPException(status_code=401, detail="Incorrect username or password")
        
    access_token = auth.create_access_token(
        data={"sub": db_user.username}, expires_delta=timedelta(minutes=auth.ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/api/login/voice")
async def login_voice(
    username: str = Form(...),
    audio: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    db_user = db.query(models.User).filter(models.User.username == username).first()
    if not db_user:
        raise HTTPException(status_code=401, detail="User not found")
        
    # 保存登录音频
    audio_path = os.path.join(UPLOAD_DIR, f"{username}_login.webm")
    with open(audio_path, "wb") as buffer:
        shutil.copyfileobj(audio.file, buffer)
        
    try:
        login_embedding = vp_engine.extract_feature(audio_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Audio processing failed: {str(e)}")
        
    # 获取注册的声纹
    if not db_user.voiceprint_embedding:
        raise HTTPException(status_code=400, detail="User has no registered voiceprint")
        
    reg_embedding = json.loads(db_user.voiceprint_embedding)
    
    # 计算相似度
    similarity = vp_engine.compare(login_embedding, reg_embedding)
    
    # 设定阈值
    # 在 CN-Celeb + 0dB 噪声扩充重训后，本地 test1(异人)=0.35、test2(同人)=0.71。
    # 阈值上调到 0.60，兼顾安全性与通过率。
    THRESHOLD = 0.60
    
    if similarity >= THRESHOLD:
        access_token = auth.create_access_token(
            data={"sub": db_user.username}, expires_delta=timedelta(minutes=auth.ACCESS_TOKEN_EXPIRE_MINUTES)
        )
        return {
            "access_token": access_token, 
            "token_type": "bearer",
            "similarity": similarity,
            "status": "success"
        }
    else:
        raise HTTPException(
            status_code=401, 
            detail=f"Voiceprint mismatch (Similarity: {similarity:.2f})"
        )

# 挂载前端静态文件 (假设后续我们把前端放到 frontend 目录)
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# 提供静态文件服务
app.mount("/static", StaticFiles(directory="frontend"), name="static")

@app.get("/")
async def root():
    return FileResponse("frontend/index.html")
