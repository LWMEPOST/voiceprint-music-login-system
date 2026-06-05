import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ArcFace(nn.Module):
    """
    ArcFace margin layer: 增大同类特征的紧凑性，扩大异类特征的间距。
    公式: cos(theta + m)
    """
    def __init__(self, in_features, out_features, s=30.0, m=0.50):
        super(ArcFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label=None):
        # 归一化权重和输入特征
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        if label is None:
            # 推理阶段，直接返回余弦相似度（缩放后）
            return cosine * self.s
            
        # 训练阶段
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        
        # 处理角度超出 pi 的情况
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        
        # 只在对应的正确类别上应用 margin
        one_hot = torch.zeros(cosine.size(), device=input.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output

class LightweightVoiceprintCNN(nn.Module):
    """
    轻量化声纹识别网络 (支持 ArcFace 度量学习)
    输入: (Batch, 1, n_mels, max_len) 比如 (B, 1, 64, 128)
    输出: (Batch, embedding_dim) 默认 128 维
    """
    def __init__(self, num_classes, embedding_dim=128, use_arcface=True):
        super(LightweightVoiceprintCNN, self).__init__()
        self.use_arcface = use_arcface
        
        # 1. 深度可分离卷积块 1
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(2, 2)  # (B, 16, 32, 64)
        
        # 2. 深度可分离卷积块 2
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, groups=16)
        self.conv2_pointwise = nn.Conv2d(32, 32, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(2, 2)  # (B, 32, 16, 32)
        
        # 3. 深度可分离卷积块 3
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, groups=32)
        self.conv3_pointwise = nn.Conv2d(64, 64, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(2, 2)  # (B, 64, 8, 16)
        
        # 4. 全局平均池化 (Global Average Pooling) -> 时频不变性
        self.gap = nn.AdaptiveAvgPool2d((1, 1)) # (B, 64, 1, 1)
        
        # 5. Embedding 层
        self.fc_embed = nn.Linear(64, embedding_dim)
        
        # 6. 分类层 (用于训练)
        if self.use_arcface:
            self.classifier = ArcFace(in_features=embedding_dim, out_features=num_classes)
        else:
            self.classifier = nn.Linear(embedding_dim, num_classes)
        
    def forward(self, x, label=None, return_embedding=False):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = F.relu(self.bn2(self.conv2_pointwise(x)))
        x = self.pool2(x)
        
        x = self.conv3(x)
        x = F.relu(self.bn3(self.conv3_pointwise(x)))
        x = self.pool3(x)
        
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        
        embed = self.fc_embed(x)
        # 归一化特征，有利于余弦相似度计算
        embed = F.normalize(embed, p=2, dim=1)
        
        if return_embedding:
            return embed
            
        if self.use_arcface:
            out = self.classifier(embed, label)
        else:
            out = self.classifier(embed)
            
        return out
