import os
import numpy as np
import onnxruntime as ort
import librosa
import noisereduce as nr
import soundfile as sf
import io
import subprocess
import tempfile
from fastdtw import fastdtw
from scipy.spatial.distance import cosine

class VoiceprintEngine:
    def __init__(self, model_path="models/weights/voiceprint_model.onnx"):
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        
    def extract_feature(self, audio_path, sr=16000, n_mels=64, max_len=128):
        try:
            # Android 录制的 webm 本质上可能是包含 opus 或 vorbis 编码的容器。
            # soundfile 在新版本中通过 libsndfile 支持部分 ogg/vorbis，但不一定完美支持 webm
            # 我们先尝试使用 soundfile 读取
            data, samplerate = sf.read(audio_path)
            if len(data.shape) > 1:
                data = data.mean(axis=1) # 转单声道
            if samplerate != sr:
                y = librosa.resample(data, orig_sr=samplerate, target_sr=sr)
            else:
                y = data
        except Exception as e1:
            try:
                # 如果 soundfile 失败，直接调用本地 ffmpeg 强行转换
                ffmpeg_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "bin", "ffmpeg.exe"))
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
                    temp_path = temp_wav.name
                
                # 执行 ffmpeg 转换，屏蔽不必要的输出
                subprocess.run([
                    ffmpeg_path, "-y", "-i", audio_path, 
                    "-ac", "1", "-ar", str(sr), temp_path
                ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
                
                y, _ = librosa.load(temp_path, sr=sr, mono=True)
                os.remove(temp_path)
            except Exception as e2:
                # 终极后备：尝试直接给 librosa 强读
                try:
                    y, _ = librosa.load(audio_path, sr=sr, mono=True)
                except Exception as e3:
                    raise Exception(f"Audio decoding failed. \nsoundfile: {e1} \nffmpeg subprocess: {e2} \nlibrosa: {e3}")
        
        # 0. 绝对音量检查：判断整个音频的峰值是否过低（纯底噪）
        max_amp = np.max(np.abs(y))
        if max_amp < 0.015:  # 经验值：正常说话峰值通常大于 0.1，0.015 以下基本是静音或微弱底噪
            raise Exception("No voice detected (volume is too low, empty microphone).")

        # 1. VAD (基于绝对能量的静音切除)
        y_trim, index = librosa.effects.trim(y, top_db=35, ref=1.0)
        
        # 核心防欺骗检查：如果切除静音后，剩余音频太短（比如不到 0.3 秒），说明根本没有说话！
        if len(y_trim) < int(sr * 0.3):
            raise Exception("No voice detected (audio too short after removing silence).")
            
        # 计算音频有效部分的均方根能量 (RMS)，判断是否音量过小
        rms = librosa.feature.rms(y=y_trim)
        if np.mean(rms) < 0.005:  # 提高均方根能量的判定阈值
            raise Exception("No voice detected (average volume too low).")
        
        # 2. 降噪
        # 使用“原始音频开头片段”作为噪声样本，避免把已裁剪后的语音当成噪声误消除
        # （之前使用 y_trim[:0.3s] 在用户开口很快时会误伤人声，导致同人比对不稳定）
        lead_len = int(sr * 0.2)
        noise_clip = y[:lead_len] if len(y) >= lead_len else y
        noise_clip = np.asarray(noise_clip, dtype=np.float32)
        speech_rms = np.sqrt(np.mean(np.square(y_trim)) + 1e-10)
        noise_rms = np.sqrt(np.mean(np.square(noise_clip)) + 1e-10) if len(noise_clip) > 0 else 1.0
        if len(noise_clip) >= int(sr * 0.05) and noise_rms < speech_rms * 0.6:
            y_noise_reduce = nr.reduce_noise(y=y_trim, y_noise=noise_clip, sr=sr, prop_decrease=0.8)
        else:
            y_noise_reduce = y_trim
            
        # 3. 预加重
        y_pre = librosa.effects.preemphasis(y_noise_reduce)
        
        # ==========================================
        # 新增：提取传统的 MFCC 特征用于 DTW 时序比对
        # MFCC 保留了更多的人声物理共振峰特征，抗信道干扰能力更强
        # ==========================================
        mfcc = librosa.feature.mfcc(y=y_pre, sr=sr, n_mfcc=20, hop_length=512)
        # 对 MFCC 进行倒谱均值方差归一化 (CMVN) 以增强跨设备一致性
        mfcc_mean = np.mean(mfcc, axis=1, keepdims=True)
        mfcc_std = np.std(mfcc, axis=1, keepdims=True)
        mfcc = (mfcc - mfcc_mean) / (mfcc_std + 1e-6)
        
        # 4. 提取梅尔频谱用于 CNN 模型
        S = librosa.feature.melspectrogram(y=y_pre, sr=sr, n_mels=n_mels, fmax=8000, hop_length=512)
        S_dB = librosa.power_to_db(S, ref=np.max)
        
        # 5. 截断或填充
        if S_dB.shape[1] > max_len:
            start = (S_dB.shape[1] - max_len) // 2
            S_dB = S_dB[:, start:start + max_len]
        else:
            pad_width = max_len - S_dB.shape[1]
            # 关键修复：使用确定性填充，避免随机噪声导致同一音频多次提特征结果漂移
            pad_value = float(np.min(S_dB)) if S_dB.size > 0 else -80.0
            pad_block = np.full((n_mels, pad_width), pad_value, dtype=S_dB.dtype)
            S_dB = np.concatenate([S_dB, pad_block], axis=1)
            
        mean = np.mean(S_dB)
        std = np.std(S_dB)
        S_dB = (S_dB - mean) / (std + 1e-5)
        
        # ONNX 推理获取 CNN Embedding
        input_tensor = np.expand_dims(np.expand_dims(S_dB, axis=0), axis=0).astype(np.float32)
        cnn_embedding = self.session.run(None, {self.input_name: input_tensor})[0][0]
        
        # 返回一个组合特征字典
        return {
            "cnn_embedding": cnn_embedding.tolist(),
            "mfcc_sequence": mfcc.T.tolist()  # 转置为 (时间帧, 特征维) 供 DTW 使用
        }
        
    @staticmethod
    def _safe_cosine_distance(vec1, vec2):
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 < 1e-8 or norm2 < 1e-8:
            return 1.0
        return float(cosine(vec1, vec2))

    def compare(self, feat1, feat2):
        """综合计算相似度：融合 CNN 余弦相似度 和 MFCC DTW 距离"""
        # 1. 计算 CNN 余弦相似度
        embed1 = np.array(feat1["cnn_embedding"])
        embed2 = np.array(feat2["cnn_embedding"])
        dot_product = np.dot(embed1, embed2)
        norm_a = np.linalg.norm(embed1)
        norm_b = np.linalg.norm(embed2)
        cnn_sim = 0.0
        if norm_a != 0 and norm_b != 0:
            cnn_sim = float(dot_product / (norm_a * norm_b))
            
        # 2. 计算 MFCC 的 DTW 距离（使用逐帧余弦距离，更稳健）
        mfcc1 = np.array(feat1["mfcc_sequence"])
        mfcc2 = np.array(feat2["mfcc_sequence"])
        
        # DTW 计算两段变长时序特征的对齐距离
        distance, path = fastdtw(mfcc1, mfcc2, dist=self._safe_cosine_distance)
        
        # 归一化 DTW 距离（除以对齐路径长度）
        normalized_dtw_dist = distance / max(len(path), 1)
        
        # 将 DTW 距离映射为相似度（Sigmoid 映射，增强中间区间可分性）
        # 经验上 0.62 左右是“同/异人”边界区域，scale 控制过渡陡峭度
        dtw_sim = 1.0 / (1.0 + np.exp((normalized_dtw_dist - 0.62) / 0.045))
        
        print(f"Debug -> CNN Sim: {cnn_sim:.4f}, DTW Dist: {normalized_dtw_dist:.4f}, DTW Sim: {dtw_sim:.4f}")
        
        # 3. 综合得分：弱化单独 CNN 的影响，强化时序特征匹配
        final_score = (cnn_sim * 0.30) + (dtw_sim * 0.70)
        
        # 反“特征坍塌”惩罚：CNN 极高但 DTW 明显不匹配时，下调最终分
        if cnn_sim > 0.90 and dtw_sim < 0.45:
            final_score -= 0.12
        
        return float(np.clip(final_score, 0.0, 1.0))

engine = VoiceprintEngine()
