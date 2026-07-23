import os
import sys
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision.ops import box_iou, batched_nms

# ==========================================
# 🚨 Bulletproof Path Resolution
# ==========================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dataloaders.kitti_dataset import KittiMultitaskDataset, multitask_collate_fn
from models.model import HydraNetMultitaskModel

try:
    from scripts.ss.evaluate_ss import compute_global_iou
    HAS_SS_METRICS = True
except ImportError:
    HAS_SS_METRICS = False

try:
    from scripts.de.evaluate_de import compute_depth_metrics
    HAS_DE_METRICS = True
except ImportError:
    HAS_DE_METRICS = False

def parse_args():
    parser = argparse.ArgumentParser(description="HydraNet 3-Task Unified Evaluation")
    parser.add_argument('--data_root', type=str, default=os.path.join(PROJECT_ROOT, "dummy_data"),
                        help='Root directory containing task dataset folders')
    parser.add_argument('--weights', type=str, default=os.path.join(PROJECT_ROOT, 'checkpoints/runs/best_multitask_model.pth'))
    parser.add_argument('--split', type=str, default='test', help='Dataset split to evaluate (val/test)')
    parser.add_argument('--conf_thresh', type=float, default=0.01, help='Confidence threshold for mAP evaluation')
    return parser.parse_args()

def decode_yolo_dfl_multiclass(preds, img_size=(192, 640), conf_thresh=0.01):
    if preds is None: return torch.empty((0,4)), torch.empty(0), torch.empty(0)
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
    args = parse_args()
    print("="*60)
    print("🚀 Initiating HydraNet 3-Task Unified Evaluation")
    print(f"📊 Evaluating Split: {args.split.upper()}")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = HydraNetMultitaskModel(num_ss_classes=7, num_od_classes=7).to(device)
    if not os.path.exists(args.weights):
        raise FileNotFoundError(f"❌ Weights not found at {args.weights}.")
        
    checkpoint = torch.load(args.weights, map_location=device)
    model.load_state_dict(checkpoint.get('model_state_dict', checkpoint), strict=False)
    model.eval()

    # 🔥 确保 Dataloader 使用正确的 kwargs 传递新的目录名称
    # 假设你的 KittiMultitaskDataset 已经支持或者能够智能推断 'semantic' 目录
    # 如果它内部还是写死的 'labels'，你需要去 dataset.py 里把它也改掉。
    val_loader = DataLoader(
        KittiMultitaskDataset(data_root=args.data_root, split=args.split), 
        batch_size=1,  
        shuffle=False, 
        collate_fn=multitask_collate_fn
    )

    all_ss_preds, all_ss_targets = [], []
    all_de_preds, all_de_targets = [], []
    
    all_od_preds = []
    all_od_gts = {}
    total_od_gt = 0

    print("\nRunning forward passes and collecting metrics...")
    with torch.no_grad():
        for batch_idx, (images, od_targets, ss_masks, de_masks) in enumerate(val_loader):
            images = images.to(device)
            out_od, out_ss, out_de = model(images)
            
            # --- 🛡️ 任务间解耦：使用独立且安全的 Try-Except 块包裹 ---
            
            # --- Task 1: Semantic Segmentation ---
            try:
                if out_ss is not None and ss_masks is not None:
                    pred_ss = torch.argmax(out_ss, dim=1).cpu().numpy()
                    target_ss = ss_masks.numpy()
                    for p, t in zip(pred_ss, target_ss):
                        if (t != 255).any():
                            all_ss_preds.append(p)
                            all_ss_targets.append(t)
            except Exception as e:
                print(f"\n⚠️ SS Evaluation skipped for batch {batch_idx}: {e}")

            # --- Task 2: Depth Estimation ---
            try:
                if out_de is not None and de_masks is not None:
                    pred_de = out_de.squeeze(1).cpu().numpy()
                    target_de = de_masks.numpy()
                    for p, t in zip(pred_de, target_de):
                        if (t > 0).any():
                            all_de_preds.append(p)
                            all_de_targets.append(t)
            except Exception as e:
                print(f"\n⚠️ DE Evaluation skipped for batch {batch_idx}: {e}")

            # --- Task 3: Object Detection (Absolute Bulletproof) ---
            try:
                if out_od is not None and od_targets is not None:
                    # 1. Parse Ground Truths 
                    if isinstance(od_targets, (list, tuple)):
                        target_tensor = od_targets[0] if len(od_targets) > 0 else torch.empty((0, 5))
                    elif od_targets.ndim == 3:
                        target_tensor = od_targets[0]
                    elif od_targets.ndim == 2 and od_targets.shape[1] == 6:
                        target_tensor = od_targets[od_targets[:, 0] == 0][:, 1:]
                    elif od_targets.ndim == 2 and od_targets.shape[1] == 5:
                        target_tensor = od_targets
                    else:
                        target_tensor = torch.empty((0, 5))
                        
                    if len(target_tensor) > 0:
                        cls_ids = target_tensor[:, 0]
                        cx, cy, w, h = target_tensor[:, 1], target_tensor[:, 2], target_tensor[:, 3], target_tensor[:, 4]
                        x1, y1, x2, y2 = (cx - w/2)*640, (cy - h/2)*192, (cx + w/2)*640, (cy + h/2)*192
                        gt_boxes = torch.stack([x1, y1, x2, y2], dim=-1)
                        gt_data = torch.cat([gt_boxes, cls_ids.unsqueeze(-1)], dim=-1)
                        total_od_gt += len(gt_boxes)
                    else:
                        gt_data = torch.empty((0, 5))
                        
                    all_od_gts[batch_idx] = gt_data.cpu()

                    # 2. Decode Predictions
                    boxes, scores, class_ids = decode_yolo_dfl_multiclass(out_od, conf_thresh=args.conf_thresh)
                    
                    if len(boxes) > 0:
                        keep = batched_nms(boxes, scores, class_ids, iou_threshold=0.45)
                        for i in keep:
                            all_od_preds.append({
                                'img_idx': batch_idx,
                                'box': boxes[i].cpu(),
                                'score': scores[i].item(),
                                'class_id': int(class_ids[i].item())
                            })
            except Exception as e:
                print(f"\n⚠️ OD Evaluation skipped for batch {batch_idx}: {e}")

            sys.stdout.write(f"\r  👉 Processed batch {batch_idx+1}/{len(val_loader)}")
            sys.stdout.flush()

    print("\n\nComputing Final Metrics...")
    
    # 1. Compute SS Metrics
    mIoU = 0.0
    if HAS_SS_METRICS and len(all_ss_preds) > 0:
        class_ious = compute_global_iou(np.stack(all_ss_preds), np.stack(all_ss_targets), num_classes=7)
        valid_ious = [iou for iou in class_ious if not np.isnan(iou)]
        mIoU = np.mean(valid_ious) * 100.0 if valid_ious else 0.0

    # 2. Compute DE Metrics
    abs_rel, rmse, a1 = 0.0, 0.0, 0.0
    if HAS_DE_METRICS and len(all_de_preds) > 0:
        abs_rel, rmse, mae, a1, a2, a3 = compute_depth_metrics(np.stack(all_de_preds), np.stack(all_de_targets))
    
    # 3. Compute OD mAP Metrics
    final_map_50_95, final_map_50 = 0.0, 0.0
    if len(all_od_preds) > 0 and len(all_od_gts) > 0:
        all_od_preds.sort(key=lambda x: x['score'], reverse=True)
        thresholds = np.linspace(0.5, 0.95, 10)
        aps = []

        for iou_thresh in thresholds:
            tp = np.zeros(len(all_od_preds))
            fp = np.zeros(len(all_od_preds))
            gt_matched = {k: np.zeros(len(v), dtype=bool) for k, v in all_od_gts.items()}

            for i, pred in enumerate(all_od_preds):
                img_idx = pred['img_idx']
                # Safeguard in case gt_data for this img_idx was never added
                gt_data = all_od_gts.get(img_idx, torch.empty((0, 5))) 
                
                if len(gt_data) == 0:
                    fp[i] = 1
                    continue
                
                gt_boxes = gt_data[:, :4]
                gt_classes = gt_data[:, 4]
                
                ious = box_iou(pred['box'].unsqueeze(0), gt_boxes).squeeze(0)
                
                best_iou = 0.0
                best_idx = -1
                for g_i in range(len(gt_boxes)):
                    if int(gt_classes[g_i].item()) == int(pred['class_id']) and ious[g_i] > best_iou:
                        best_iou = ious[g_i].item()
                        best_idx = g_i
                
                if best_iou >= iou_thresh and best_idx != -1 and not gt_matched[img_idx][best_idx]:
                    tp[i] = 1
                    gt_matched[img_idx][best_idx] = True
                else:
                    fp[i] = 1

            if total_od_gt > 0:
                tp_cumsum = np.cumsum(tp)
                fp_cumsum = np.cumsum(fp)
                recalls = tp_cumsum / total_od_gt
                precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-16)
                aps.append(compute_ap(recalls, precisions))
            else:
                aps.append(0.0)

        if len(aps) > 0:
            final_map_50_95 = float(np.mean(aps)) * 100.0
            final_map_50 = float(aps[0]) * 100.0

    # ==========================================
    # 🔥 Boss Report Generation
    # ==========================================
    print("\n" + "🔥"*12 + " HydraNet 3-Task Boss Report " + "🔥"*12)
    
    print(f"🎯 [Object Detection] (Evaluated {len(all_od_gts)} samples)")
    print(f"     - mAP@[0.5:0.95] : {final_map_50_95:.2f}%")
    print(f"     - mAP@0.50       : {final_map_50:.2f}%")
    print("-" * 65)
    
    print(f"🎨 [Semantic Segmentation] (Evaluated {len(all_ss_preds)} valid images)")
    if HAS_SS_METRICS:
        print(f"     - mIoU           : {mIoU:.2f}%")
    else:
        print(f"     - mIoU           : [Missing evaluate_ss module]")
    print("-" * 65)
    
    print(f"📏 [Depth Estimation] (Evaluated {len(all_de_preds)} valid images)")
    if HAS_DE_METRICS:
        print(f"     - AbsRel         : {abs_rel:.4f}  ↓ (Lower is better)")
        print(f"     - RMSE           : {rmse:.4f}  ↓")
        print(f"     - Acc <1.25      : {a1*100:.2f}%  ↑ (Higher is better)")
    else:
        print(f"     - Metrics        : [Missing evaluate_de module]")
        
    print("="*65)
    print("🎉 Ultimate Multi-Task Evaluation Finished Successfully!")

if __name__ == "__main__":
    main()