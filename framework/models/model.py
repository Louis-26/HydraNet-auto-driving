import os
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2
from torch.nn import functional as F
from typing import cast
from models.encoder import MobileNetV2Encoder
from models.decoder import LightWeightRefineNet


# ==========================================
# 2. Feature Fusion Neck: RefineNet (CRP and Fusion only)
# ==========================================
class CRPBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn_in = nn.BatchNorm2d(out_channels)
        
        # Simplified CRP chained pooling
        self.pool = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
        self.conv_out = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        x = F.relu(self.bn_in(self.conv_in(x)))
        out = x
        pooled = self.conv_out(self.pool(x))
        return out + pooled

class RefineNetNeck(nn.Module):
    def __init__(self, refine_dim=256):
        super().__init__()
        
        # Dimensionality reduction mapping to align channels
        self.adapt4 = nn.Conv2d(320, refine_dim, 1)
        self.adapt3 = nn.Conv2d(96, refine_dim, 1)
        self.adapt2 = nn.Conv2d(32, refine_dim, 1)
        
        self.crp4 = CRPBlock(refine_dim, refine_dim)
        self.crp3 = CRPBlock(refine_dim, refine_dim)
        self.crp2 = CRPBlock(refine_dim, refine_dim)

    def forward(self, l1, l2, l3, l4):
        # Top-down fusion pathway
        L7 = self.crp4(self.adapt4(l4)) 
        
        x3 = self.adapt3(l3)
        L7_up = F.interpolate(L7, size=x3.size()[2:], mode='bilinear', align_corners=False)
        L5 = self.crp3(x3 + L7_up)      
        
        x2 = self.adapt2(l2)
        L5_up = F.interpolate(L5, size=x2.size()[2:], mode='bilinear', align_corners=False)
        L3 = self.crp2(x2 + L5_up)      
        
        return L3, L5, L7

# ==========================================
# 3. Object Detection Decoder: YOLOv8 Decoupled Head
# ==========================================
class YOLOv8DetectionHead(nn.Module):
    def __init__(self, in_channels=256, num_classes=1, reg_max=16):
        super().__init__()
        self.num_classes = num_classes
        self.reg_max = reg_max
        
        # Create three independent detection modules for L3, L5, L7
        self.heads = nn.ModuleList()
        for _ in range(3):
            # Dimensionality reduction and compression using Conv + MaxPool
            stem = nn.Sequential(
                nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1, stride=2),
                nn.BatchNorm2d(in_channels // 2),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
            
            # Decoupled Bounding Box branch
            bbox_branch = nn.Sequential(
                nn.Conv2d(in_channels // 2, in_channels // 2, 3, padding=1),
                nn.BatchNorm2d(in_channels // 2),
                nn.ReLU(),
                nn.Conv2d(in_channels // 2, 4 * reg_max, 1) 
            )
            
            # Decoupled Class branch
            cls_branch = nn.Sequential(
                nn.Conv2d(in_channels // 2, in_channels // 2, 3, padding=1),
                nn.BatchNorm2d(in_channels // 2),
                nn.ReLU(),
                nn.Conv2d(in_channels // 2, num_classes, 1)
            )
            
            self.heads.append(nn.ModuleDict({
                'stem': stem,
                'bbox': bbox_branch,
                'cls': cls_branch
            }))

    def forward(self, L3, L5, L7):
        features = [L3, L5, L7]
        outputs = []
        
        for i, feature in enumerate(features):
            # Use 'cast' to resolve Pylance strict type checking issues elegantly
            head_dict = cast(nn.ModuleDict, self.heads[i])
            
            x = head_dict['stem'](feature)
            bbox_pred = head_dict['bbox'](x)
            cls_pred = head_dict['cls'](x)
            
            outputs.append({
                'bbox': bbox_pred, 
                'cls': cls_pred
            })
            
        return outputs

# ==========================================
# 4. Assembled Test Model: HydraNet (Detection Branch Only)
# ==========================================
class HydraNetDetectionModel(nn.Module):
    def __init__(self, num_classes=1): 
        super().__init__()
        self.encoder = MobileNetV2Encoder(pretrained=True)
        self.neck = RefineNetNeck(refine_dim=256)
        self.yolo_head = YOLOv8DetectionHead(in_channels=256, num_classes=num_classes)

    def forward(self, x):
        l1, l2, l3, l4 = self.encoder(x)
        L3, L5, L7 = self.neck(l1, l2, l3, l4)
        det_outputs = self.yolo_head(L3, L5, L7)
        return det_outputs

# Add this class to your existing model.py

class HydraNetDepthModel(nn.Module):
    def __init__(self):
        """
        Depth Estimation branch of HydraNet.
        Architecture: MobileNetV2 (Encoder) + Light-Weight RefineNet (Decoder).
        """
        super().__init__()
        
        # 1. Shared Encoder Backbone
        self.encoder = MobileNetV2Encoder(pretrained=True)
        
        # 2. Task-Specific Decoder for Depth (Output is a single continuous channel)
        self.decoder = LightWeightRefineNet(num_classes=1)

    def forward(self, x):
        features = self.encoder(x)
        
        if isinstance(features, (list, tuple)):
            raw_features = features[-1]
        elif isinstance(features, dict):
            raw_features = list(features.values())[-1]
        else:
            raw_features = features
            
        out = self.decoder(raw_features)
        out = F.interpolate(out, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False)
        
        # Depth strictly cannot be negative. Apply ReLU to enforce positive predictions.
        out = F.relu(out)
        
        return out

    
class HydraNetSegmentationModel(nn.Module):
    def __init__(self, num_classes=7):
        """
        Semantic Segmentation branch of HydraNet.
        Architecture: MobileNetV2 (Encoder) + Light-Weight RefineNet (Decoder).
        """
        super().__init__()
        
        # 1. Shared Encoder Backbone
        self.encoder = MobileNetV2Encoder(pretrained=True)
        
        # 2. Task-Specific Decoder (Light-Weight RefineNet)
        # MobileNetV2 final feature output typically has 1280 channels
        self.decoder = LightWeightRefineNet(num_classes=num_classes)

    def forward(self, x):
        # Extract features from the encoder backbone
        features = self.encoder(x)
        
        # Extract the deepest feature representation
        if isinstance(features, (list, tuple)):
            raw_features = features[-1]
        elif isinstance(features, dict):
            raw_features = list(features.values())[-1]
        else:
            raw_features = features
            
        # Refine features through Light-Weight RefineNet
        out = self.decoder(raw_features)
        
        # Upsample back to the original image input resolution (e.g., 192, 640)
        out = F.interpolate(out, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False)
        
        return out
    


class HydraNetMultitaskModel(nn.Module):
    def __init__(self, num_ss_classes=7, num_od_classes=7):
        """
        The Ultimate HydraNet Multi-Task Architecture (3-Heads).
        1 Shared Encoder + 3 Independent Decoders (OD, SS, DE).
        """
        super().__init__()
        # 1. Shared Encoder Backbone
        self.encoder = MobileNetV2Encoder(pretrained=True)
        
        # ==========================================
        # Branch 1: Object Detection (OD)
        # Fully implemented Neck and YOLOv8 Head
        # ==========================================
        self.od_neck = RefineNetNeck(refine_dim=256)
        self.od_head = YOLOv8DetectionHead(in_channels=256, num_classes=num_od_classes)
        
        # ==========================================
        # Branch 2: Semantic Segmentation (SS)
        # ==========================================
        self.ss_decoder = LightWeightRefineNet(num_classes=num_ss_classes)
        
        # ==========================================
        # Branch 3: Depth Estimation (DE)
        # ==========================================
        self.de_decoder = LightWeightRefineNet(num_classes=1)

    def forward(self, x):
        # 1. Shared Encoder Feature Extraction
        l1, l2, l3, l4 = self.encoder(x)
        
        # 2. OD Branch (Uses the specialized Neck and YOLO Head)
        L3, L5, L7 = self.od_neck(l1, l2, l3, l4)
        out_od = self.od_head(L3, L5, L7)
        
        # 3. SS Branch (Uses the deepest feature map l4)
        out_ss = self.ss_decoder(l4)
        out_ss = F.interpolate(out_ss, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False)
        
        # 4. DE Branch (Uses the deepest feature map l4)
        out_de = self.de_decoder(l4)
        out_de = F.interpolate(out_de, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False)
        out_de = F.relu(out_de) # Depth strictly non-negative
        
        return out_od, out_ss, out_de


    
if __name__ == "__main__":
    # Simulate an image tensor with KITTI dimensions
    dummy_image = torch.randn(2, 3, 192, 640)
    
    model = HydraNetDetectionModel(num_classes=1)
    preds = model(dummy_image)
    
    print("✅ Model forward pass successful!")
    for idx, pred in enumerate(preds):
        print(f"Scale {idx} - BBox Shape: {pred['bbox'].shape}, Class Shape: {pred['cls'].shape}")