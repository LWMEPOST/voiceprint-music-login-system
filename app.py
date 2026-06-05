#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
音乐平台声纹登录系统 - 手机适配版（带嘈杂环境降噪）
功能：声纹注册 + 声纹登录 + 音频录制 + MFCC声纹比对 + 手机端适配 + 嘈杂环境人声提取
"""
import streamlit as st
import os
import wave
import pyaudio
import numpy as np
import json
import librosa
import noisereduce as nr  # 降噪库

# ======================== 基础配置 ========================
# 音频参数
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 5  # 录音时长
AUDIO_DIR = "./audio_data"
USER_DATA_FILE = "./user_data.json"

# 创建必要目录
os.makedirs(AUDIO_DIR, exist_ok=True)

# ======================== 核心工具函数 ========================
def init_user_data():
    if not os.path.exists(USER_DATA_FILE):
        with open(USER_DATA_FILE, "w", encoding="utf-8") as f:
            json.dump({}, f)

def get_user_data():
    init_user_data()
    with open(USER_DATA_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def save_user_data(user_data):
    with open(USER_DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(user_data, f, ensure_ascii=False, indent=4)

def extract_audio_feature(audio_path):
    """提取MFCC声纹特征（带嘈杂环境降噪+人声提取）"""
    try:
        y, sr = librosa.load(audio_path, sr=RATE)
        # 静音段切除
        y_trim, _ = librosa.effects.trim(y, top_db=20)
        # 专业降噪
        noise_clip = y_trim[:int(sr*0.3)]
        y_noise_reduce = nr.reduce_noise(y=y_trim, y_noise=noise_clip, sr=sr, prop_decrease=0.85)
        # 预加重，提升人声高频特征
        y_pre = librosa.effects.preemphasis(y_noise_reduce)
        # 提取MFCC特征
        mfcc = librosa.feature.mfcc(y=y_pre, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1)
        return mfcc_mean.tolist()
    except Exception as e:
        st.error(f"音频特征提取失败：{str(e)}")
        return None

def compare_audio_feature(feature1, feature2):
    """比对声纹特征（欧氏距离+归一化相似度）"""
    if not feature1 or not feature2:
        return 0.0
    distance = np.linalg.norm(np.array(feature1) - np.array(feature2))
    similarity = max(0.0, 100.0 - min(distance / 2, 100.0))
    return similarity

# ======================== 录音函数 ========================
def record_audio(username, is_register=True):
    """录音函数"""
    try:
        st.info(f"🎤 开始录音（{RECORD_SECONDS}秒），请说话...")
        p = pyaudio.PyAudio()

        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)

        frames = []
        for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK)
            frames.append(data)

        stream.stop_stream()
        stream.close()
        p.terminate()

        audio_type = "register" if is_register else "login"
        audio_path = os.path.join(AUDIO_DIR, f"{username}_{audio_type}.wav")
        
        wf = wave.open(audio_path, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()

        st.success("✅ 录音完成！")
        return audio_path
    except Exception as e:
        st.error(f"❌ 录音失败：{str(e)}")
        return None

# ======================== 注册/登录核心逻辑 ========================
def register_user(username, password, confirm_pwd):
    """注册逻辑"""
    if not username or not password:
        return "❌ 用户名/密码不能为空！"
    if password != confirm_pwd:
        return "❌ 两次密码不一致！"
    
    user_data = get_user_data()
    if username in user_data:
        return "❌ 用户名已存在！"
    
    audio_path = record_audio(username, is_register=True)
    if not audio_path:
        return "❌ 注册失败：录音异常！"
    
    feature = extract_audio_feature(audio_path)
    if not feature:
        return "❌ 注册失败：声纹特征提取失败！"
    
    user_data[username] = {
        "password": password,
        "audio_feature": feature,
        "audio_path": audio_path
    }
    save_user_data(user_data)
    
    return f"✅ 注册成功！欢迎 {username}，请前往登录。"

def login_user(username, password):
    """登录逻辑"""
    if not username or not password:
        return "❌ 用户名/密码不能为空！"
    
    user_data = get_user_data()
    if username not in user_data:
        return "❌ 用户名不存在！"
    
    if user_data[username]["password"] != password:
        return "❌ 密码错误！"
    
    audio_path = record_audio(username, is_register=False)
    if not audio_path:
        return "❌ 登录失败：录音异常！"
    
    login_feature = extract_audio_feature(audio_path)
    register_feature = user_data[username]["audio_feature"]
    similarity = compare_audio_feature(login_feature, register_feature)
    
    if similarity >= 30:
        return f"🎉 登录成功！声纹相似度：{similarity:.2f}%（匹配）"
    else:
        return f"⚠️  登录失败！声纹相似度：{similarity:.2f}%（不匹配）"

# ======================== Streamlit 页面布局（手机适配版） ========================
def main():
    st.set_page_config(
        page_title="🎵 音乐平台声纹登录",
        page_icon="🎵",
        layout="centered"
    )

    # ===== 手机端适配样式（核心优化代码）=====
    st.markdown("""
        <style>
        /* 全局手机适配：去除边距，适配竖屏 */
        .main {padding: 0 10px !important; background-color: #f8f9fa;}
        /* 手机端按钮：占满宽度，方便点击 */
        .stButton>button {
            width: 100% !important;
            height: 50px !important;
            font-size: 16px !important;
            border-radius: 8px;
            background-color: #007bff;
            color: white;
        }
        /* 手机端输入框：放大字体和内边距 */
        .stTextInput>div>div>input {
            font-size: 16px !important;
            padding: 12px !important;
            border-radius: 8px;
        }
        /* 标签页适配手机：缩小间距，放大字体 */
        .stTabs [data-baseweb="tab"] {
            font-size: 15px !important;
            height: 45px !important;
        }
        /* 标题适配手机 */
        h1 {font-size: 24px !important; text-align: center;}
        h2 {font-size: 20px !important; margin: 15px 0 !important;}
        h5 {font-size: 15px !important; text-align: center;}
        /* 提示文字适配 */
        .stCaption {font-size: 14px !important; line-height: 1.5;}
        </style>
    """, unsafe_allow_html=True)
    # ==========================================

    st.title("🎵 音乐平台声纹登录系统")
    st.divider()

    tab1, tab2 = st.tabs(["📱 声纹注册", "🔑 声纹登录"])

    with tab1:
        st.subheader("账号注册")
        reg_username = st.text_input("用户名", placeholder="请输入昵称/手机号", key="reg_name")
        reg_password = st.text_input("密码", type="password", placeholder="请设置密码", key="reg_pwd")
        reg_confirm_pwd = st.text_input("确认密码", type="password", placeholder="再次输入密码", key="reg_cpwd")
        
        if st.button("🎤 录制声纹并注册", key="reg_btn", type="primary"):
            result = register_user(reg_username, reg_password, reg_confirm_pwd)
            st.markdown(f"<h5>{result}</h5>", unsafe_allow_html=True)

    with tab2:
        st.subheader("声纹登录")
        login_username = st.text_input("用户名", placeholder="请输入注册的用户名", key="login_name")
        login_password = st.text_input("密码", type="password", placeholder="请输入密码", key="login_pwd")
        
        if st.button("🎤 录制声纹并登录", key="login_btn", type="primary"):
            result = login_user(login_username, login_password)
            st.markdown(f"<h5>{result}</h5>", unsafe_allow_html=True)

    st.divider()
    st.caption("💡 提示：支持嘈杂环境录音（风扇/车流/轻微说话声），手机访问请确保与电脑在同一WiFi下。")

if __name__ == "__main__":
    main()