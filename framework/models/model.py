import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2
import torch.nn.functional as F

# ==========================================
# 1. 共享主干: MobileNetV2 Encoder[cite: 1]
# ==========================================
class MobileNetV2Encoder(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        # 截取预训练 MobileNetV2 的不同 stage
        features = mobilenet_v2(pretrained=pretrained).features
        self.layer1 = features[:4]   # Output channels: 24
        self.layer2 = features[4:7]  # Output channels: 32
        self.layer3 = features[7:14] # Output channels: 96
        self.layer4 = features[14:]  # Output channels: 320

    def forward(self, x):
        l1 = self.layer1(x)
        l2 = self.layer2(l1)
        l3 = self.layer3(l2)
        l4 = self.layer4(l3)
        return l1, l2, l3, l4

# ==========================================
# 2. 特征融合颈部: RefineNet (仅保留 CRP 和特征融合部分)[cite: 1]
# ==========================================
class CRPBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn_in = nn.BatchNorm2d(out_channels)
        # 简化版 CRP 链式池化
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
        # 降维映射
        self.adapt4 = nn.Conv2d(320, refine_dim, 1)
        self.adapt3 = nn.Conv2d(96, refine_dim, 1)
        self.adapt2 = nn.Conv2d(32, refine_dim, 1)
        
        self.crp4 = CRPBlock(refine_dim, refine_dim)
        self.crp3 = CRPBlock(refine_dim, refine_dim)
        self.crp2 = CRPBlock(refine_dim, refine_dim)

    def forward(self, l1, l2, l3, l4):
        # 对应报告图示，自顶向下融合
        L7 = self.crp4(self.adapt4(l4)) # 深层
        
        x3 = self.adapt3(l3)
        L7_up = F.interpolate(L7, size=x3.size()[2:], mode='bilinear', align_corners=False)
        L5 = self.crp3(x3 + L7_up)      # 中层
        
        x2 = self.adapt2(l2)
        L5_up = F.interpolate(L5, size=x2.size()[2:], mode='bilinear', align_corners=False)
        L3 = self.crp2(x2 + L5_up)      # 浅层
        
        return L3, L5, L7

# ==========================================
# 3. 目标检测解码器: YOLOv8 Decoupled Head[cite: 1]
# ==========================================
class YOLOv8DetectionHead(nn.Module):
    def __init__(self, in_channels=256, num_classes=1, reg_max=16):
        super().__init__()
        self.num_classes = num_classes
        self.reg_max = reg_max
        
        # 为 L3, L5, L7 创建三个独立的检测模块
        self.heads = nn.ModuleList()
        for _ in range(3):
            # 报告中提到：特征图输入后经过自定义的 Conv + MaxPool 降维压缩[cite: 1]
            stem = nn.Sequential(
                nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1, stride=2),
                nn.BatchNorm2d(in_channels // 2),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
            
            # 分离的 Bounding Box 分支[cite: 1]
            bbox_branch = nn.Sequential(
                nn.Conv2d(in_channels // 2, in_channels // 2, 3, padding=1),
                nn.BatchNorm2d(in_channels // 2),
                nn.ReLU(),
                nn.Conv2d(in_channels // 2, 4 * reg_max, 1) # 预测分布，用于 DFL[cite: 1]
            )
            
            # 分离的 Class 分支[cite: 1]
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
            # 1. 降维
            x = self.heads[i]['stem'](feature)
            # 2. 分支预测
            bbox_pred = self.heads[i]['bbox'](x)
            cls_pred = self.heads[i]['cls'](x)
            
            outputs.append({
                'bbox': bbox_pred, 
                'cls': cls_pred
            })
            
        return outputs

# ==========================================
# 4. 组装测试模型: HydraNet (仅激活 Detection 分支)
# ==========================================
class HydraNetDetectionModel(nn.Module):
    def __init__(self, num_classes=1): # 跑通阶段，假设只检测 Car (1类)
        super().__init__()
        self.encoder = MobileNetV2Encoder(pretrained=True)
        self.neck = RefineNetNeck(refine_dim=256)
        self.yolo_head = YOLOv8DetectionHead(in_channels=256, num_classes=num_classes)

    def forward(self, x):
        # 1. 提取基础特征[cite: 1]
        l1, l2, l3, l4 = self.encoder(x)
        
        # 2. RefineNet 构造特征金字塔[cite: 1]
        L3, L5, L7 = self.neck(l1, l2, l3, l4)
        
        # 3. YOLOv8 解码器输出检测结果[cite: 1]
        det_outputs = self.yolo_head(L3, L5, L7)
        
        return det_outputs

# 测试代码是否畅通
if __name__ == "__main__":
    # 模拟一张 KITTI 尺寸的图片 (Batch_Size=2, Channels=3, Height=192, Width=640)[cite: 1]
    dummy_image = torch.randn(2, 3, 192, 640)
    
    model = HydraNetDetectionModel(num_classes=1)
    preds = model(dummy_image)
    
    print("模型前向传播成功！")
    for idx, pred in enumerate(preds):
        print(f"Scale {idx} - BBox Shape: {pred['bbox'].shape}, Class Shape: {pred['cls'].shape}")