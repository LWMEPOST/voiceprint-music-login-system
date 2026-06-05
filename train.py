import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from models.network import LightweightVoiceprintCNN
import matplotlib.pyplot as plt

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

class VoiceprintDataset(Dataset):
    def __init__(self, features, labels):
        # features shape: (N, 64, 128)
        self.features = torch.tensor(features, dtype=torch.float32).unsqueeze(1) # (N, 1, 64, 128)
        self.labels = torch.tensor(labels, dtype=torch.long)
        
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def train_model():
    print("Loading robust dataset...")
    X = np.load("dataset/features_robust/features.npy")
    y = np.load("dataset/features_robust/labels.npy")
    num_classes = len(np.unique(y))
    print(f"Loaded {len(X)} samples, {num_classes} classes.")
    
    # 划分训练集和测试集 (8:2)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    train_dataset = VoiceprintDataset(X_train, y_train)
    test_dataset = VoiceprintDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 启用 ArcFace 以拉开类间距离
    model = LightweightVoiceprintCNN(num_classes=num_classes, embedding_dim=128, use_arcface=True).to(device)
    
    criterion = nn.CrossEntropyLoss()
    # 针对 ArcFace 适当降低学习率，加入权重衰减
    optimizer = optim.Adam(model.parameters(), lr=0.0008, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)
    
    num_epochs = 25
    best_acc = 0.0
    no_improve_epochs = 0
    
    train_losses = []
    val_accs = []
    
    os.makedirs("models/weights", exist_ok=True)
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            # ArcFace 需要传入 label
            outputs = model(inputs, label=labels)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)
        
        # 验证
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                # 验证时不需要 label（只返回相似度做预测分类）
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        val_acc = 100 * correct / total
        val_accs.append(val_acc)
        scheduler.step(val_acc)
        
        print(
            f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, "
            f"Val Acc: {val_acc:.2f}%, LR: {optimizer.param_groups[0]['lr']:.6f}"
        )
        
        if val_acc >= best_acc:
            best_acc = val_acc
            no_improve_epochs = 0
            torch.save(model.state_dict(), "models/weights/best_model.pth")
            print("--> Best model saved.")
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= 8:
                print("Early stopping triggered.")
                break
            
    print("Training complete. Best Validation Accuracy:", best_acc)
    
    # 绘制训练曲线
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title("Training Loss")
    plt.subplot(1, 2, 2)
    plt.plot(val_accs)
    plt.title("Validation Accuracy")
    plt.savefig("models/weights/training_curve.png")
    
    # 导出 ONNX (优化部署)
    print("Exporting ONNX model...")
    model.load_state_dict(torch.load("models/weights/best_model.pth", weights_only=True))
    model.eval()
    
    # 临时修改 forward 以便只返回 embedding (绕过分类层)
    # 为了干净导出，我们可以包装一下
    class ONNXWrapper(nn.Module):
        def __init__(self, core_model):
            super().__init__()
            self.core = core_model
        def forward(self, x):
            return self.core(x, return_embedding=True)
            
    export_model = ONNXWrapper(model)
    
    dummy_input = torch.randn(1, 1, 64, 128).to(device)
    
    # 强制将所有数据打包在单一 onnx 文件中，防止生成 .data 分离文件导致路径报错
    # 并且使用绝对路径避免 Windows 路径分隔符问题
    onnx_path = os.path.abspath("models/weights/voiceprint_model.onnx")
    
    # 使用 fallback (旧版) 导出 API 避免 torch 2.5 下 onnx 导出工具因 external_data 导致的路径错误
    # fallback 参数让底层放弃使用实验性的 dynamo exporter 而是用传统 JIT
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(
            export_model, dummy_input, onnx_path,
            export_params=True, opset_version=18, do_constant_folding=True,
            input_names=['input'], output_names=['embedding'],
            dynamic_axes={'input': {0: 'batch_size'}, 'embedding': {0: 'batch_size'}},
            external_data=False
        )
    print(f"ONNX model exported to {onnx_path}")

if __name__ == "__main__":
    train_model()
