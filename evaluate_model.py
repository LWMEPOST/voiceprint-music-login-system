import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import seaborn as sns
from models.network import LightweightVoiceprintCNN

def evaluate_model():
    print("Loading data...")
    X = np.load("dataset/features_robust/features.npy")
    y = np.load("dataset/features_robust/labels.npy")
    num_classes = len(np.unique(y))
    
    device = torch.device("cpu")
    model = LightweightVoiceprintCNN(num_classes=num_classes, embedding_dim=128, use_arcface=True).to(device)
    model.load_state_dict(torch.load("models/weights/best_model.pth", map_location=device, weights_only=True))
    model.eval()

    print("Extracting embeddings...")
    embeddings = []
    with torch.no_grad():
        for i in range(len(X)):
            # X[i] shape is (64, 128) -> make it (1, 1, 64, 128)
            input_tensor = torch.tensor(X[i], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            emb = model(input_tensor, return_embedding=True)
            embeddings.append(emb.numpy()[0])
            
    embeddings = np.array(embeddings)
    
    # Calculate all pairwise cosine similarities
    print("Calculating pairwise similarities...")
    same_class_sims = []
    diff_class_sims = []
    
    # Normalize embeddings just in case (they should be already by the network)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / (norms + 1e-10)
    
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            sim = np.dot(embeddings[i], embeddings[j])
            if y[i] == y[j]:
                same_class_sims.append(sim)
            else:
                diff_class_sims.append(sim)
                
    same_class_sims = np.array(same_class_sims)
    diff_class_sims = np.array(diff_class_sims)
    
    print(f"Same class pairs: {len(same_class_sims)}")
    print(f"Diff class pairs: {len(diff_class_sims)}")
    
    print(f"Same class sim mean: {np.mean(same_class_sims):.4f}, std: {np.std(same_class_sims):.4f}")
    print(f"Diff class sim mean: {np.mean(diff_class_sims):.4f}, std: {np.std(diff_class_sims):.4f}")
    
    # Prepare data for ROC curve
    y_true = np.concatenate([np.ones_like(same_class_sims), np.zeros_like(diff_class_sims)])
    y_scores = np.concatenate([same_class_sims, diff_class_sims])
    
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    # Calculate EER (Equal Error Rate)
    fnr = 1 - tpr
    eer_threshold = thresholds[np.nanargmin(np.absolute((fnr - fpr)))]
    eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    print(f"EER (Equal Error Rate): {eer*100:.2f}% at threshold {eer_threshold:.4f}")
    
    # Plotting
    os.makedirs("docs", exist_ok=True)
    
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Similarity Distribution
    plt.subplot(1, 2, 1)
    sns.histplot(same_class_sims, color='green', label='Same Person (Genuine)', kde=True, stat='density', bins=50)
    sns.histplot(diff_class_sims, color='red', label='Different Person (Impostor)', kde=True, stat='density', bins=50)
    plt.axvline(x=eer_threshold, color='black', linestyle='--', label=f'EER Threshold ({eer_threshold:.2f})')
    plt.title('Cosine Similarity Distribution')
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Density')
    plt.legend()
    
    # Plot 2: ROC Curve
    plt.subplot(1, 2, 2)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.plot(eer, 1-eer, marker='o', markersize=8, color="red", label=f'EER = {eer*100:.2f}%')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    
    plt.tight_layout()
    save_path = "docs/model_evaluation.png"
    plt.savefig(save_path, dpi=300)
    print(f"Evaluation plots saved to {save_path}")

if __name__ == "__main__":
    evaluate_model()
