import os
import sys
import json
from datetime import datetime
import torch
import numpy as np
import torchvision
from torch.utils.data import DataLoader
from torchvision.ops import box_iou, batched_nms

# ==========================================
# 🚨 Bulletproof Path Resolution
# ==========================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dataloaders.kitti_dataset import KittiDetectionDataset, kitti_collate_fn
from models.model import HydraNetDetectionModel

def save_od_metrics(metrics_dict, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    json_path = os.path.join(out_dir, "mAP_summary.json")
    txt_path = os.path.join(out_dir, "mAP_summary.txt")
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(metrics_dict, f, indent=4)
        
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("=" * 45 + "\n")
        f.write("HydraNet Object Detection Evaluation Summary\n")
        f.write(f"Generated at: {timestamp}\n")
        f.write("=" * 45 + "\n")
        f.write(f"Final mAP@[0.5:0.95] : {metrics_dict.get('mAP@[0.5:0.95]', 0.0):.4f}%\n")
        f.write(f"mAP@0.5              : {metrics_dict.get('mAP@0.5', 0.0):.4f}%\n")
        f.write(f"mAP@0.75             : {metrics_dict.get('mAP@0.75', 0.0):.4f}%\n")
        f.write("=" * 45 + "\n")

    print(f"\n[INFO] mAP results successfully saved to:")
    print(f"  - {json_path}")
    print(f"  - {txt_path}")

def decode_yolo_dfl_multiclass(preds, img_size=(192, 640), conf_thresh=0.01):
    if preds is None:
        return torch.empty((0,4)), torch.empty(0), torch.empty(0)
        
    all_bboxes, all_scores, all_class_ids = [], [], []
    reg_max = 16 
    dfl_weights = torch.arange(reg_max, dtype=torch.float32, device=preds[0]['bbox'].device)
    img_h, img_w = img_size

    for pred in preds:
        bbox_feat, cls_feat = pred['bbox'], pred['cls']
        B, C, grid_h, grid_w = cls_feat.shape
        stride_h, stride_w = img_h / grid_h, img_w / grid_w
        
        cls_scores = torch.sigmoid(cls_feat).squeeze(0)
        max_scores, max_class_ids = torch.max(cls_scores, dim=0)
        
        bbox_feat = bbox_feat.squeeze(0).view(4, reg_max, grid_h, grid_w).permute(2, 3, 0, 1)
        bbox_feat = torch.nn.functional.softmax(bbox_feat, dim=-1)
        dist = (bbox_feat * dfl_weights).sum(dim=-1)
        
        y_coords, x_coords = torch.meshgrid(torch.arange(grid_h), torch.arange(grid_w), indexing='ij')
        y_coords = (y_coords.float().to(bbox_feat.device) + 0.5) * stride_h
        x_coords = (x_coords.float().to(bbox_feat.device) + 0.5) * stride_w
        
        x1 = x_coords - dist[..., 0] * stride_w
        y1 = y_coords - dist[..., 1] * stride_h
        x2 = x_coords + dist[..., 2] * stride_w
        y2 = y_coords + dist[..., 3] * stride_h
        
        bboxes = torch.stack([x1, y1, x2, y2], dim=-1).view(-1, 4)
        scores = max_scores.view(-1)
        class_ids = max_class_ids.view(-1)
        
        mask = scores > conf_thresh
        if mask.sum() > 0:
            all_bboxes.append(bboxes[mask])
            all_scores.append(scores[mask])
            all_class_ids.append(class_ids[mask])
            
    if len(all_bboxes) == 0:
        return torch.empty((0,4)), torch.empty(0), torch.empty(0)
        
    return torch.cat(all_bboxes), torch.cat(all_scores), torch.cat(all_class_ids)

def compute_ap(recall, precision):
    if len(recall) == 0: return 0.0
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def main():
    print("====================================================")
    print("🚀 Initiating COCO-Style mAP@[0.5:0.95] Evaluation (Single OD)")
    print("====================================================")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    val_img_dir = os.path.join(PROJECT_ROOT, "dummy_data", "od", "train", "images")
    val_lbl_dir = os.path.join(PROJECT_ROOT, "dummy_data", "od", "train", "YOLO_labels")
    
    val_dataset = KittiDetectionDataset(image_dir=val_img_dir, label_dir=val_lbl_dir, target_size=(192, 640))
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=kitti_collate_fn)

    model = HydraNetDetectionModel(num_classes=7).to(device)
    
    weight_path = os.path.join(PROJECT_ROOT, "checkpoints", "runs", "best_detection_model.pth")
    if not os.path.exists(weight_path):
        alt_weight = os.path.join(PROJECT_ROOT, "checkpoints", "runs", "best_multitask_model.pth")
        if os.path.exists(alt_weight):
            weight_path = alt_weight
        else:
            raise FileNotFoundError("No valid weights found. Train the model first!")
        
    model.load_state_dict(torch.load(weight_path, map_location=device)['model_state_dict'], strict=False)
    model.eval()

    all_preds = []
    all_gts = {}
    total_gt = 0

    print("Running Inference on Validation Set...")
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(val_dataloader):
            images = images.to(device)
            
            # ========================================================
            # 🛡️ 终极防弹 Ground Truth 解析器 (彻底解决标签被吞的问题)
            # ========================================================
            if isinstance(targets, (list, tuple)):
                target_tensor = targets[0] if len(targets) > 0 else torch.empty((0, 5))
            elif targets.ndim == 3:
                target_tensor = targets[0]
            elif targets.ndim == 2 and targets.shape[1] == 6:
                target_tensor = targets[targets[:, 0] == 0][:, 1:]
            elif targets.ndim == 2 and targets.shape[1] == 5:
                target_tensor = targets  # [class_id, cx, cy, w, h]
            else:
                target_tensor = torch.empty((0, 5))
                
            if len(target_tensor) > 0:
                cls_ids = target_tensor[:, 0]
                cx, cy, w, h = target_tensor[:, 1], target_tensor[:, 2], target_tensor[:, 3], target_tensor[:, 4]
                x1, y1, x2, y2 = (cx - w/2)*640, (cy - h/2)*192, (cx + w/2)*640, (cy + h/2)*192
                gt_boxes = torch.stack([x1, y1, x2, y2], dim=-1)
                gt_data = torch.cat([gt_boxes, cls_ids.unsqueeze(-1)], dim=-1)
                total_gt += len(gt_boxes)
            else:
                gt_boxes = torch.empty((0, 4))
                cls_ids = torch.empty(0)
                gt_data = torch.empty((0, 5))
                
            all_gts[batch_idx] = gt_data.cpu()

            # ========================================================
            # 🚨 模式切换区 (验证代码绝对正确后，请解除注释恢复正常预测)
            # ========================================================
            # 【正常预测模式】
            preds = model(images)
            boxes, scores, class_ids = decode_yolo_dfl_multiclass(preds, conf_thresh=0.01)
            
            # 【上帝模式 - 作弊验证】(直接使用标签当预测框)
            # if len(gt_boxes) > 0:
            #     boxes = gt_boxes.to(device)
            #     scores = torch.ones(len(gt_boxes), device=device)
            #     class_ids = cls_ids.to(device)
            # else:
            #     boxes = torch.empty((0, 4), device=device)
            #     scores = torch.empty(0, device=device)
            #     class_ids = torch.empty(0, device=device)
            # ========================================================
            
            if len(boxes) > 0:
                keep = batched_nms(boxes, scores, class_ids, iou_threshold=0.45)
                for i in keep:
                    all_preds.append({
                        'img_idx': batch_idx,
                        'box': boxes[i].cpu(),
                        'score': scores[i].item(),
                        'class_id': int(class_ids[i].item())
                    })

            sys.stdout.write(f"\r  👉 Processed image {batch_idx+1}/{len(val_dataloader)}")
            sys.stdout.flush()

    all_preds.sort(key=lambda x: x['score'], reverse=True)
    thresholds = np.linspace(0.5, 0.95, 10)
    aps = []

    print(f"\n\nCalculating mAP across {len(thresholds)} thresholds...")
    for iou_thresh in thresholds:
        tp = np.zeros(len(all_preds))
        fp = np.zeros(len(all_preds))
        gt_matched = {k: np.zeros(len(v), dtype=bool) for k, v in all_gts.items()}

        for i, pred in enumerate(all_preds):
            img_idx = pred['img_idx']
            gt_data = all_gts[img_idx]
            if len(gt_data) == 0:
                fp[i] = 1
                continue
            
            gt_boxes = gt_data[:, :4]
            gt_classes = gt_data[:, 4]
            
            ious = box_iou(pred['box'].unsqueeze(0), gt_boxes).squeeze(0)
            
            best_iou = 0.0
            best_idx = -1
            for g_i in range(len(gt_boxes)):
                # 强转 int 防止张量类型对比失败
                if int(gt_classes[g_i].item()) == int(pred['class_id']) and ious[g_i] > best_iou:
                    best_iou = ious[g_i].item()
                    best_idx = g_i
            
            if best_iou >= iou_thresh and best_idx != -1 and not gt_matched[img_idx][best_idx]:
                tp[i] = 1
                gt_matched[img_idx][best_idx] = True
            else:
                fp[i] = 1

        if total_gt == 0:
            aps.append(0.0)
            continue
            
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        recalls = tp_cumsum / total_gt
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-16)
        aps.append(compute_ap(recalls, precisions))

    final_map_50_95 = float(np.mean(aps)) if len(aps) > 0 else 0.0
    final_map_50 = float(aps[0]) if len(aps) > 0 else 0.0
    final_map_75 = float(aps[5]) if len(aps) > 5 else 0.0

    print("====================================================")
    print(f"Final mAP@[0.5:0.95] : {final_map_50_95*100:.2f}%")
    print(f"mAP@0.5              : {final_map_50*100:.2f}%")
    print(f"mAP@0.75             : {final_map_75*100:.2f}%")
    print("====================================================")

    metrics_dict = {
        "mAP@[0.5:0.95]": final_map_50_95 * 100.0,
        "mAP@0.5": final_map_50 * 100.0,
        "mAP@0.75": final_map_75 * 100.0
    }
    output_dir = os.path.join(PROJECT_ROOT, "outputs", "dummy", "od", "test")
    save_od_metrics(metrics_dict, output_dir)

if __name__ == "__main__":
    main()