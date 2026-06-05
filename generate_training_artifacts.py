import os
import json
import glob
import datetime
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import librosa.display
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.model_selection import train_test_split

from models.network import LightweightVoiceprintCNN


def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _batch_iter(x: np.ndarray, batch_size: int = 256):
    n = len(x)
    for i in range(0, n, batch_size):
        yield x[i:i + batch_size]


def find_visualization_wav():
    """
    为波形图与频谱图寻找一个可用的示例音频。
    优先使用早期合成数据集 dataset/audio，其次回退到真实测试集目录，
    这样在新电脑上即使没有生成旧版合成数据，也能继续产出分析图。
    """
    candidate_patterns = [
        ("synthetic", os.path.abspath("dataset/audio/*/*.wav"), False),
        ("real_fsdd", os.path.abspath("dataset/audio_real/*/*.wav"), False),
        ("real_chinese", os.path.abspath("dataset/audio_real_chinese/*/*.wav"), False),
        ("cnceleb", os.path.abspath("dataset/cn-celeb-test/**/*.wav"), True),
    ]

    for source_name, pattern, recursive in candidate_patterns:
        wav_candidates = sorted(glob.glob(pattern, recursive=recursive))
        if wav_candidates:
            return wav_candidates[0], source_name

    searched = "\n".join(f"- {pattern}" for _, pattern, _ in candidate_patterns)
    raise FileNotFoundError(
        "未找到可用于波形图/频谱图的音频文件。已检查路径：\n"
        f"{searched}\n"
        "请先准备 dataset/audio、dataset/audio_real、dataset/audio_real_chinese "
        "或 dataset/cn-celeb-test 中的任意一类音频。"
    )


def extract_embeddings(model, x_np, device, batch_size=256):
    model.eval()
    embs = []
    with torch.no_grad():
        for batch in _batch_iter(x_np, batch_size):
            inp = torch.tensor(batch, dtype=torch.float32, device=device).unsqueeze(1)
            emb = model(inp, return_embedding=True).cpu().numpy()
            embs.append(emb)
    return np.concatenate(embs, axis=0)


def evaluate_cls(model, x_np, y_np, device, batch_size=256):
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    all_preds = []
    criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    with torch.no_grad():
        for i in range(0, len(x_np), batch_size):
            xb = torch.tensor(x_np[i:i + batch_size], dtype=torch.float32, device=device).unsqueeze(1)
            yb = torch.tensor(y_np[i:i + batch_size], dtype=torch.long, device=device)
            logits = model(xb)
            loss = criterion(logits, yb)
            preds = torch.argmax(logits, dim=1)
            total_loss += loss.item()
            total += yb.size(0)
            correct += (preds == yb).sum().item()
            all_preds.append(preds.cpu().numpy())

    avg_loss = total_loss / max(total, 1)
    acc = correct / max(total, 1)
    y_pred = np.concatenate(all_preds)
    return avg_loss, acc, y_pred


def plot_waveform_and_spectrogram(audio_path: str, output_dir: str):
    y, sr = librosa.load(audio_path, sr=None)
    t = np.arange(len(y)) / sr

    waveform_path = os.path.join(output_dir, "waveform.png")
    plt.figure(figsize=(12, 4))
    plt.plot(t, y, linewidth=0.8)
    plt.title(f"Waveform: {os.path.basename(audio_path)}")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.savefig(waveform_path, dpi=220)
    plt.close()

    spectrogram_path = os.path.join(output_dir, "spectrogram.png")
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64, fmax=8000)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mel_db, sr=sr, x_axis="time", y_axis="mel", cmap="magma")
    plt.colorbar(format="%+2.0f dB")
    plt.title(f"Mel Spectrogram: {os.path.basename(audio_path)}")
    plt.tight_layout()
    plt.savefig(spectrogram_path, dpi=220)
    plt.close()

    return waveform_path, spectrogram_path


def plot_grad_cam(model, sample_np: np.ndarray, output_dir: str, device):
    activations = {}
    gradients = {}

    def fw_hook(_, __, out):
        activations["value"] = out.detach()

    def bw_hook(_, grad_in, grad_out):
        gradients["value"] = grad_out[0].detach()

    h1 = model.conv3_pointwise.register_forward_hook(fw_hook)
    h2 = model.conv3_pointwise.register_full_backward_hook(bw_hook)

    x = torch.tensor(sample_np, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    x.requires_grad_(True)
    model.eval()
    logits = model(x)
    cls_idx = int(torch.argmax(logits, dim=1).item())
    score = logits[0, cls_idx]

    model.zero_grad(set_to_none=True)
    score.backward()

    acts = activations["value"]  # (1, C, H, W)
    grads = gradients["value"]   # (1, C, H, W)
    weights = grads.mean(dim=(2, 3), keepdim=True)
    cam = (weights * acts).sum(dim=1, keepdim=True)
    cam = F.relu(cam)
    cam = F.interpolate(cam, size=(sample_np.shape[0], sample_np.shape[1]), mode="bilinear", align_corners=False)
    cam = cam.squeeze().cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

    cam_path = os.path.join(output_dir, "cam_heatmap.png")
    plt.figure(figsize=(10, 4))
    plt.imshow(sample_np, origin="lower", aspect="auto", cmap="gray")
    plt.imshow(cam, origin="lower", aspect="auto", cmap="jet", alpha=0.45)
    plt.colorbar(label="CAM intensity")
    plt.title(f"Grad-CAM (predicted class={cls_idx})")
    plt.xlabel("Time frames")
    plt.ylabel("Mel bins")
    plt.tight_layout()
    plt.savefig(cam_path, dpi=220)
    plt.close()

    h1.remove()
    h2.remove()
    return cam_path, cls_idx


def compute_feature_importance_by_occlusion(model, x_np, y_np, device, output_dir, subset_size=400):
    rng = np.random.default_rng(42)
    if len(x_np) > subset_size:
        idx = rng.choice(len(x_np), size=subset_size, replace=False)
        xs = x_np[idx]
        ys = y_np[idx]
    else:
        xs = x_np
        ys = y_np

    base_loss, _, _ = evaluate_cls(model, xs, ys, device, batch_size=256)
    n_mels = xs.shape[1]
    importance = np.zeros(n_mels, dtype=np.float32)
    fill_value = float(xs.mean())

    for mel_idx in range(n_mels):
        x_occ = xs.copy()
        x_occ[:, mel_idx, :] = fill_value
        loss_i, _, _ = evaluate_cls(model, x_occ, ys, device, batch_size=256)
        importance[mel_idx] = loss_i - base_loss

    fig_path = os.path.join(output_dir, "feature_importance.png")
    plt.figure(figsize=(12, 4))
    plt.bar(np.arange(n_mels), importance, width=0.9)
    plt.title("Feature Importance (Mel-bin Occlusion, loss increase)")
    plt.xlabel("Mel bin index")
    plt.ylabel("Importance")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=220)
    plt.close()

    top_bins = np.argsort(-importance)[:10].tolist()
    return fig_path, top_bins, importance


def main():
    np.random.seed(42)
    torch.manual_seed(42)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = _ensure_dir(os.path.abspath(os.path.join("artifacts", f"training_eval_{timestamp}")))

    features_path = os.path.abspath("dataset/features_robust/features.npy")
    labels_path = os.path.abspath("dataset/features_robust/labels.npy")
    model_path = os.path.abspath("models/weights/best_model.pth")

    X = np.load(features_path)
    y = np.load(labels_path)
    num_classes = int(len(np.unique(y)))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LightweightVoiceprintCNN(num_classes=num_classes, embedding_dim=128, use_arcface=True).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    # 1) Loss + Accuracy + confusion matrix
    test_loss, test_acc, y_pred = evaluate_cls(model, X_test, y_test, device, batch_size=256)

    cm = confusion_matrix(y_test, y_pred, labels=np.arange(num_classes), normalize="true")
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, cmap="Blues", cbar=True, square=False, xticklabels=False, yticklabels=False)
    plt.title("Normalized Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(cm_path, dpi=220)
    plt.close()

    loss_acc_path = os.path.join(output_dir, "loss_accuracy.png")
    plt.figure(figsize=(8, 4))
    plt.bar(["Test Loss", "Test Accuracy"], [test_loss, test_acc * 100.0], color=["#1f77b4", "#ff7f0e"])
    plt.title("Current Model Performance")
    plt.ylabel("Value (Loss / %)")
    plt.tight_layout()
    plt.savefig(loss_acc_path, dpi=220)
    plt.close()

    # 2) Verification scores -> ROC / DET / EER / minDCF
    train_emb = extract_embeddings(model, X_train, device, batch_size=256)
    test_emb = extract_embeddings(model, X_test, device, batch_size=256)

    # 每类中心模板
    centroids = np.zeros((num_classes, train_emb.shape[1]), dtype=np.float32)
    for c in range(num_classes):
        cent = train_emb[y_train == c].mean(axis=0)
        cent = cent / (np.linalg.norm(cent) + 1e-10)
        centroids[c] = cent

    test_emb = test_emb / (np.linalg.norm(test_emb, axis=1, keepdims=True) + 1e-10)
    score_mat = test_emb @ centroids.T  # (N_test, C)
    genuine_scores = score_mat[np.arange(len(y_test)), y_test]

    mask = np.ones_like(score_mat, dtype=bool)
    mask[np.arange(len(y_test)), y_test] = False
    impostor_scores = score_mat[mask]

    y_true = np.concatenate(
        [np.ones_like(genuine_scores, dtype=np.int32), np.zeros_like(impostor_scores, dtype=np.int32)]
    )
    y_scores = np.concatenate([genuine_scores, impostor_scores])

    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1.0 - tpr
    roc_auc = float(auc(fpr, tpr))
    eer_idx = int(np.nanargmin(np.abs(fnr - fpr)))
    eer = float((fpr[eer_idx] + fnr[eer_idx]) / 2.0)
    eer_threshold = float(thresholds[eer_idx])

    # minDCF
    p_target = 0.01
    c_miss = 1.0
    c_fa = 1.0
    c_det = c_miss * fnr * p_target + c_fa * fpr * (1.0 - p_target)
    c_def = min(c_miss * p_target, c_fa * (1.0 - p_target))
    c_det_norm = c_det / (c_def + 1e-12)
    mindcf_idx = int(np.argmin(c_det_norm))
    min_dcf = float(c_det_norm[mindcf_idx])
    mindcf_threshold = float(thresholds[mindcf_idx])

    roc_path = os.path.join(output_dir, "roc_curve.png")
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"ROC (AUC={roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.scatter([fpr[eer_idx]], [tpr[eer_idx]], color="red", label=f"EER={eer * 100:.2f}%")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(roc_path, dpi=220)
    plt.close()

    det_path = os.path.join(output_dir, "det_curve.png")
    plt.figure(figsize=(6, 6))
    plt.plot(fpr * 100.0, fnr * 100.0)
    plt.scatter([fpr[eer_idx] * 100.0], [fnr[eer_idx] * 100.0], color="red", label=f"EER={eer * 100:.2f}%")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("FAR (%)")
    plt.ylabel("FRR (%)")
    plt.title("DET Curve")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(det_path, dpi=220)
    plt.close()

    eer_path = os.path.join(output_dir, "eer_curve.png")
    plt.figure(figsize=(8, 4))
    plt.plot(thresholds, fpr, label="FPR")
    plt.plot(thresholds, fnr, label="FNR")
    plt.axvline(eer_threshold, linestyle="--", color="red", label=f"EER thr={eer_threshold:.4f}")
    plt.scatter([eer_threshold], [eer], color="red")
    plt.xlabel("Threshold")
    plt.ylabel("Error Rate")
    plt.title("EER Analysis")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(eer_path, dpi=220)
    plt.close()

    mindcf_path = os.path.join(output_dir, "mindcf_curve.png")
    plt.figure(figsize=(8, 4))
    plt.plot(thresholds, c_det_norm, label="Normalized DCF")
    plt.axvline(mindcf_threshold, linestyle="--", color="red", label=f"minDCF={min_dcf:.4f}")
    plt.scatter([mindcf_threshold], [min_dcf], color="red")
    plt.xlabel("Threshold")
    plt.ylabel("Normalized DCF")
    plt.title("minDCF Analysis")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(mindcf_path, dpi=220)
    plt.close()

    # 3) Feature Importance
    feat_imp_path, top_bins, importance = compute_feature_importance_by_occlusion(
        model, X_test, y_test, device, output_dir, subset_size=400
    )

    # 4) Waveform + Spectrogram
    chosen_wav, wav_source = find_visualization_wav()
    waveform_path, spectrogram_path = plot_waveform_and_spectrogram(chosen_wav, output_dir)

    # 5) CAM 热力图（使用一个测试特征样本）
    cam_path, cam_pred_class = plot_grad_cam(model, X_test[0], output_dir, device)

    metrics = {
        "dataset": {
            "features_path": features_path,
            "labels_path": labels_path,
            "num_samples": int(len(X)),
            "num_classes": int(num_classes),
            "train_samples": int(len(X_train)),
            "test_samples": int(len(X_test)),
        },
        "model": {
            "weights_path": model_path,
            "device": str(device),
        },
        "classification": {
            "test_loss": float(test_loss),
            "test_accuracy": float(test_acc),
        },
        "verification": {
            "auc": roc_auc,
            "eer": eer,
            "eer_threshold": eer_threshold,
            "min_dcf": min_dcf,
            "min_dcf_threshold": mindcf_threshold,
            "p_target": p_target,
            "c_miss": c_miss,
            "c_fa": c_fa,
            "num_genuine_scores": int(len(genuine_scores)),
            "num_impostor_scores": int(len(impostor_scores)),
        },
        "feature_importance": {
            "top_mel_bins": top_bins,
            "top_mel_importance": [float(importance[i]) for i in top_bins],
        },
        "visualization_input": {
            "waveform_audio_path": chosen_wav,
            "waveform_audio_source": wav_source,
            "cam_predicted_class": int(cam_pred_class),
        },
        "artifacts": {
            "roc": roc_path,
            "det": det_path,
            "eer": eer_path,
            "confusion_matrix": cm_path,
            "mindcf": mindcf_path,
            "loss_accuracy": loss_acc_path,
            "feature_importance": feat_imp_path,
            "spectrogram": spectrogram_path,
            "waveform": waveform_path,
            "cam_heatmap": cam_path,
        },
    }

    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"Artifacts saved in: {output_dir}")
    print(f"Metrics JSON: {metrics_path}")
    print(f"Test Loss: {test_loss:.6f}")
    print(f"Test Accuracy: {test_acc * 100:.2f}%")
    print(f"AUC: {roc_auc:.6f}")
    print(f"EER: {eer * 100:.4f}% (threshold={eer_threshold:.6f})")
    print(f"minDCF: {min_dcf:.6f} (threshold={mindcf_threshold:.6f})")


if __name__ == "__main__":
    main()
