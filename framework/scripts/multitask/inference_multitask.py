import os
import sys
import shutil
import argparse
import cv2
import torch
import torchvision
import numpy as np

# ==========================================
# 🚨 Bulletproof Path Resolution
# ==========================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.model import HydraNetMultitaskModel

# OD Configurations
ID_TO_CLASS = {0: "Car", 1: "Van", 2: "Truck", 3: "Pedestrian", 4: "Person_sitting", 5: "Cyclist", 6: "Tram"}
CLASS_COLORS = {0: (0, 255, 0), 1: (255, 255, 0), 2: (0, 165, 255), 3: (0, 0, 255), 4: (255, 0, 255), 5: (255, 0, 0), 6: (128, 128, 128)}

# SS Color Map
SS_COLOR_MAP = np.array([
    [128, 64, 128],  # 0: road
    [244, 35, 232],  # 1: sidewalk
    [70, 70, 70],    # 2: building
    [107, 142, 35],  # 3: vegetation
    [220, 20, 60],   # 4: person
    [0, 0, 142],     # 5: car
    [0, 0, 0]        # 6: unknown
], dtype=np.uint8)

def parse_args():
    parser = argparse.ArgumentParser(description="HydraNet 3-Task Synchronized Inference")
    parser.add_argument('--source', type=str, default=os.path.join(PROJECT_ROOT, 'dummy_data/od/test/images'))
    parser.add_argument('--weights', type=str, default=os.path.join(PROJECT_ROOT, 'checkpoints/runs/best_multitask_model.pth'))
    parser.add_argument('--out_dir', type=str, default=os.path.join(PROJECT_ROOT, 'outputs/dummy/multitask/test'))
    parser.add_argument('--conf_thresh', type=float, default=0.25)
    parser.add_argument('--iou_thresh', type=float, default=0.45)
    return parser.parse_args()

def decode_yolo_dfl(preds, img_size=(192, 640), conf_thresh=0.25):
    if preds is None: return torch.empty((0,4)), torch.empty(0), torch.empty(0)
    all_bboxes, all_scores, all_class_ids = [], [], []
    reg_max = 16 
    img_h, img_w = img_size
    for pred in preds:
        bbox_feat, cls_feat = pred['bbox'], pred['cls']
        B, C, grid_h, grid_w = cls_feat.shape
        stride_h, stride_w = img_h / grid_h, img_w / grid_w
        dfl_weights = torch.arange(reg_max, dtype=torch.float32, device=bbox_feat.device)
        cls_scores = torch.sigmoid(cls_feat).squeeze(0)
        max_scores, max_class_ids = torch.max(cls_scores, dim=0)
        
        # 🛡️ 极致防弹：加上 .squeeze(0) 防止未来 Batch size 变化时报错
        bbox_feat = pred['bbox'].squeeze(0).view(4, reg_max, grid_h, grid_w).permute(2, 3, 0, 1)
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
        scores, class_ids = max_scores.view(-1), max_class_ids.view(-1)
        mask = scores > conf_thresh
        if mask.sum() > 0:
            all_bboxes.append(bboxes[mask])
            all_scores.append(scores[mask])
            all_class_ids.append(class_ids[mask])
    if len(all_bboxes) == 0: return torch.empty((0,4)), torch.empty(0), torch.empty(0)
    return torch.cat(all_bboxes), torch.cat(all_scores), torch.cat(all_class_ids)

def main():
    args = parse_args()
    print("="*60)
    print("🚀 Initiating HydraNet 3-Task Synchronized Inference")
    print("="*60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HydraNetMultitaskModel(num_ss_classes=7, num_od_classes=7).to(device)
    if not os.path.exists(args.weights): raise FileNotFoundError(f"❌ Weights missing at {args.weights}")
    
    checkpoint = torch.load(args.weights, map_location=device)
    model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
    model.eval()

    # ==========================================
    # 🎯 严格对齐单任务结构的独立目录树
    # ==========================================
    dirs = {
        # OD
        "od_images": os.path.join(args.out_dir, "od", "images"),
        "od_annotated": os.path.join(args.out_dir, "od", "annotated_images"),
        "od_labels": os.path.join(args.out_dir, "od", "labels"),
        # SS (✨ 统一目录结构: semantic 和 semantic_rgb)
        "ss_images": os.path.join(args.out_dir, "ss", "images"),
        "ss_semantic": os.path.join(args.out_dir, "ss", "semantic"),
        "ss_semantic_rgb": os.path.join(args.out_dir, "ss", "semantic_rgb"),
        # DE
        "de_images": os.path.join(args.out_dir, "de", "images"),
        "de_depth": os.path.join(args.out_dir, "de", "depth"),
        "de_heatmaps": os.path.join(args.out_dir, "de", "heatmaps"),
        "de_pred": os.path.join(args.out_dir, "de", "predicted_depth")
    }
    
    for d in dirs.values(): 
        os.makedirs(d, exist_ok=True)

    img_files = [f for f in os.listdir(args.source) if f.endswith(('.png', '.jpg'))]

    with torch.no_grad():
        for img_name in img_files:
            img_path = os.path.join(args.source, img_name)
            base_name = os.path.splitext(img_name)[0]
            
            # 各自拷贝原图到对应的任务目录
            shutil.copy(img_path, os.path.join(dirs["od_images"], img_name))
            shutil.copy(img_path, os.path.join(dirs["ss_images"], img_name))
            shutil.copy(img_path, os.path.join(dirs["de_images"], img_name))
            
            orig_img = cv2.imread(img_path)
            orig_h, orig_w = orig_img.shape[:2]
            rgb_img = cv2.cvtColor(cv2.resize(orig_img, (640, 192)), cv2.COLOR_BGR2RGB)
            img_tensor = torch.from_numpy(rgb_img).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0

            out_od, out_ss, out_de = model(img_tensor)

            # ==========================================
            # 1. SS Inference -> ss/semantic, ss/semantic_rgb
            # ==========================================
            pred_ss_mask_raw = torch.argmax(out_ss, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
            pred_ss_mask_raw = np.clip(pred_ss_mask_raw, 0, len(SS_COLOR_MAP) - 1)
            
            # 🔥 必须用 INTER_NEAREST 保护 ID 标签不被破坏
            pred_ss_mask_resized = cv2.resize(pred_ss_mask_raw, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
            
            # A. 保存 Raw 类别 ID 图 (单通道灰度)
            cv2.imwrite(os.path.join(dirs["ss_semantic"], f"{base_name}.png"), pred_ss_mask_resized)
            
            # B. 保存彩色可视化图 (RGB)
            color_ss = SS_COLOR_MAP[pred_ss_mask_resized]
            color_ss_bgr = cv2.cvtColor(color_ss, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(dirs["ss_semantic_rgb"], f"{base_name}.png"), color_ss_bgr)

            # ==========================================
            # 2. DE Inference -> de/depth, de/heatmaps, de/predicted_depth
            # ==========================================
            pred_de_raw = out_de.squeeze().cpu().numpy()
            pred_de_resized = cv2.resize(pred_de_raw, (orig_w, orig_h))
            
            # 🛡️ 极致防弹：截断深度图负数防溢出
            pred_de_resized = np.clip(pred_de_resized, 0, None)
            
            pred_de_uint16 = (pred_de_resized * 256.0).astype(np.uint16)
            cv2.imwrite(os.path.join(dirs["de_depth"], f"{base_name}.png"), pred_de_uint16)
            cv2.imwrite(os.path.join(dirs["de_pred"], f"{base_name}.png"), pred_de_uint16)
            
            norm_depth = cv2.normalize(pred_de_resized, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            heatmap = cv2.applyColorMap(norm_depth, cv2.COLORMAP_JET)
            cv2.imwrite(os.path.join(dirs["de_heatmaps"], f"{base_name}.png"), heatmap)

            # ==========================================
            # 3. OD Inference -> od/labels, od/annotated_images
            # ==========================================
            od_visual = orig_img.copy()
            txt_path = os.path.join(dirs["od_labels"], f"{base_name}.txt")
            
            if out_od is not None:
                bboxes, scores, class_ids = decode_yolo_dfl(out_od, img_size=(192, 640), conf_thresh=args.conf_thresh)
                if len(bboxes) > 0:
                    keep_indices = torchvision.ops.batched_nms(bboxes, scores, class_ids, iou_threshold=args.iou_thresh)
                    final_bboxes, final_scores, final_classes = bboxes[keep_indices], scores[keep_indices], class_ids[keep_indices]
                    
                    with open(txt_path, "w") as f:
                        for i in range(len(final_bboxes)):
                            x1, y1, x2, y2 = final_bboxes[i].cpu().numpy()
                            score = final_scores[i].item()
                            cls_id = int(final_classes[i].item())
                            
                            x_center = np.clip(((x1 + x2) / 2) / 640, 0.0, 1.0)
                            y_center = np.clip(((y1 + y2) / 2) / 192, 0.0, 1.0)
                            w_norm = np.clip((x2 - x1) / 640, 0.0, 1.0)
                            h_norm = np.clip((y2 - y1) / 192, 0.0, 1.0)
                            f.write(f"{cls_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")

                            real_x1, real_x2 = int(x1 * (orig_w / 640)), int(x2 * (orig_w / 640))
                            real_y1, real_y2 = int(y1 * (orig_h / 192)), int(y2 * (orig_h / 192))
                            color = CLASS_COLORS.get(cls_id, (255, 255, 255))
                            
                            cv2.rectangle(od_visual, (real_x1, real_y1), (real_x2, real_y2), color, 2)
                            label_text = f"{ID_TO_CLASS.get(cls_id, 'Unknown')} ({score:.2f})"
                            cv2.putText(od_visual, label_text, (real_x1, real_y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                else:
                    open(txt_path, 'w').close()
            else:
                open(txt_path, 'w').close()
                
            cv2.imwrite(os.path.join(dirs["od_annotated"], img_name), od_visual)
            print(f"  - Synchronized outputs generated for: {img_name}")

    print("="*60)
    print(f"🎉 3-Task Inference Complete! Check structured outputs at: {args.out_dir}")

if __name__ == "__main__":
    main()