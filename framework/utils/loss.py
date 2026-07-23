import torch
import torch.nn as nn
import torch.nn.functional as F

class YOLOLoss(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.num_classes = num_classes
        
    def forward(self, predictions, targets):
        device = targets.device
        
        if targets.shape[0] == 0:
            dummy = torch.zeros(1, device=device)
            for p in predictions:
                dummy += p['cls'].sum() * 0.0 + p['bbox'].sum() * 0.0
            return dummy

        # ==========================================
        # 1. 解包 L3 (我们用最高分辨率层来精准过拟合)
        # ==========================================
        l3_preds = predictions[0] 
        pred_cls = l3_preds['cls']
        pred_bbox = l3_preds['bbox']
        
        B, C, H, W = pred_cls.shape
        
        target_cls = torch.zeros_like(pred_cls)
        target_box = torch.zeros((B, H, W, 4), device=device)
        mask = torch.zeros((B, H, W, 1), device=device)

        # ==========================================
        # 2. DFL 解码器 (坐标对齐)
        # ==========================================
        reg_max = 16
        dfl_weights = torch.arange(reg_max, dtype=torch.float32, device=device)
        bbox_feat = pred_bbox.view(B, 4, reg_max, H, W).permute(0, 3, 4, 1, 2)
        bbox_feat = F.softmax(bbox_feat, dim=-1)
        dist = (bbox_feat * dfl_weights).sum(dim=-1)
        
        y_coords, x_coords = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
        y_coords = y_coords.float() + 0.5
        x_coords = x_coords.float() + 0.5
        
        pred_x1 = (x_coords - dist[..., 0]) / W
        pred_y1 = (y_coords - dist[..., 1]) / H
        pred_x2 = (x_coords + dist[..., 2]) / W
        pred_y2 = (y_coords + dist[..., 3]) / H
        pred_boxes = torch.stack([pred_x1, pred_y1, pred_x2, pred_y2], dim=-1)

        # ==========================================
        # 3. 标签分配 
        # ==========================================
        for t in targets:
            batch_idx = int(t[0].item())
            class_id = int(t[1].item())
            cx, cy, w, h = t[2], t[3], t[4], t[5]
            
            grid_x = max(0, min(int(cx * W), W - 1))
            grid_y = max(0, min(int(cy * H), H - 1))
            
            target_cls[batch_idx, class_id, grid_y, grid_x] = 1.0
            
            gt_x1 = cx - w / 2
            gt_y1 = cy - h / 2
            gt_x2 = cx + w / 2
            gt_y2 = cy + h / 2
            
            target_box[batch_idx, grid_y, grid_x, :] = torch.tensor([gt_x1, gt_y1, gt_x2, gt_y2], device=device)
            mask[batch_idx, grid_y, grid_x, 0] = 1.0

        # ==========================================
        # 4. 🔥 致命修复：LOSS 权重重分配
        # ==========================================
        # 【修复 1: 拯救学霸】给唯一的正样本赋予 500 倍权重，防止被两万个背景淹没
        cls_loss_unreduced = F.binary_cross_entropy_with_logits(pred_cls, target_cls, reduction='none')
        pos_weight_mask = torch.ones_like(target_cls)
        pos_weight_mask[target_cls == 1.0] = 500.0  # 🔥 强行逼迫网络学懂正样本！
        loss_cls = (cls_loss_unreduced * pos_weight_mask).mean()

        # 坐标回归 Loss
        loss_box = F.l1_loss(pred_boxes * mask, target_box * mask, reduction='sum') / (mask.sum() + 1e-6)

        # 【修复 2: 暴力镇压熊孩子】强迫未经训练的 L5 和 L7 输出 0 (背景)，干掉满屏的垃圾框！
        suppress_loss = torch.zeros(1, device=device)
        for p in predictions[1:]:
            suppress_loss += F.binary_cross_entropy_with_logits(p['cls'], torch.zeros_like(p['cls']), reduction='mean')

        # 整合 Loss
        total_loss = loss_cls + 5.0 * loss_box + suppress_loss * 2.0
        return total_loss