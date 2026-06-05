import os
import glob
import time
import sys
import numpy as np
import librosa
import soundfile as sf

# 确保直接执行 tests/eval_system.py 时也能导入项目根目录下的 backend 包
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.voiceprint import VoiceprintEngine
from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d

def add_white_noise(audio_path, snr_db):
    """为音频添加指定信噪比的高斯白噪声，返回加噪后的文件路径"""
    y, sr = librosa.load(audio_path, sr=16000)
    
    # 计算信号能量
    signal_power = np.mean(y ** 2)
    
    # 根据 SNR 计算噪声能量
    # SNR = 10 * log10(P_signal / P_noise)
    noise_power = signal_power / (10 ** (snr_db / 10.0))
    
    noise = np.random.normal(0, np.sqrt(noise_power), y.shape)
    y_noisy = y + noise
    
    noisy_path = audio_path.replace(".wav", f"_noisy_{snr_db}dB.wav")
    sf.write(noisy_path, y_noisy, sr)
    return noisy_path

def evaluate_system():
    print("Initializing Voiceprint Engine...")
    engine = VoiceprintEngine("models/weights/voiceprint_model.onnx")
    
    audio_dir = "dataset/audio_real"
    speakers = sorted(os.listdir(audio_dir))
    
    # 提取每个说话人的注册模板 (取每个说话人的第 1 个样本作为注册样本)
    templates = {}
    test_samples = {}
    
    for spk in speakers:
        files = sorted(glob.glob(os.path.join(audio_dir, spk, "*.wav")))
        if len(files) < 2: continue
        
        # 提取注册特征
        templates[spk] = engine.extract_feature(files[0])
        # 剩下的作为测试样本
        test_samples[spk] = files[1:11] # 每个取 10 个做测试，以节省时间
        
    print(f"Registered {len(templates)} speakers.")
    
    # 1. 响应时间测试
    print("\n--- 1. 响应时间测试 (Response Speed) ---")
    start_time = time.time()
    for spk in speakers:
        if len(test_samples[spk]) > 0:
            engine.extract_feature(test_samples[spk][0])
    end_time = time.time()
    avg_time = (end_time - start_time) / len(speakers) * 1000
    print(f"平均特征提取与推理时间: {avg_time:.2f} ms / 条")
    
    # 2. 识别精度测试 (Clean)
    print("\n--- 2. 识别精度测试 (Clean Environment) ---")
    def compute_metrics(snr=None):
        y_true = []
        y_scores = []
        
        for spk_true in speakers:
            for sample_path in test_samples[spk_true]:
                if snr is not None:
                    sample_path = add_white_noise(sample_path, snr)
                
                feat = engine.extract_feature(sample_path)
                
                # 与所有模板比对
                for spk_ref, template in templates.items():
                    sim = engine.compare(feat, template)
                    y_scores.append(sim)
                    y_true.append(1 if spk_true == spk_ref else 0)
                    
                if snr is not None:
                    os.remove(sample_path) # 清理临时文件
                    
        fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
        eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
        return eer, fpr, tpr, thresholds
        
    eer_clean, _, _, _ = compute_metrics(snr=None)
    print(f"等错误率 (EER) - 纯净语音: {eer_clean*100:.2f}%")
    
    # 3. 抗噪性测试 (10dB, 5dB)
    print("\n--- 3. 抗噪性测试 (Noise Robustness) ---")
    eer_15db, _, _, _ = compute_metrics(snr=15)
    print(f"等错误率 (EER) - 15dB SNR: {eer_15db*100:.2f}%")
    
    eer_5db, _, _, _ = compute_metrics(snr=5)
    print(f"等错误率 (EER) - 5dB SNR: {eer_5db*100:.2f}%")
    
    print("\n测试完成。此测试仅作为该轻量化模型的基准参考，生产环境需在真实数据集上训练。")
    
if __name__ == "__main__":
    evaluate_system()
