import os
import numpy as np
import torch
import torch.nn as nn
from models.network import LightweightVoiceprintCNN
import warnings

def export_onnx():
    print("Exporting ONNX model...")
    y = np.load("dataset/features_robust/labels.npy")
    num_classes = len(np.unique(y))
    
    device = torch.device("cpu")
    model = LightweightVoiceprintCNN(num_classes=num_classes, embedding_dim=128, use_arcface=True).to(device)
    model.load_state_dict(torch.load("models/weights/best_model.pth", map_location=device, weights_only=True))
    model.eval()
    
    class ONNXWrapper(nn.Module):
        def __init__(self, core_model):
            super().__init__()
            self.core = core_model
        def forward(self, x):
            return self.core(x, return_embedding=True)
            
    export_model = ONNXWrapper(model)
    dummy_input = torch.randn(1, 1, 64, 128).to(device)
    
    onnx_path = os.path.join(os.getcwd(), "models", "weights", "voiceprint_model.onnx")
    
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
    export_onnx()
