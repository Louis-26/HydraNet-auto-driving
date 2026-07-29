import os
import sys
import argparse
import cv2
import torch
import shutil
import numpy as np
from torchvision.ops import batched_nms

# ==========================================
# Bulletproof Path Resolution
# ==========================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.model import HydraNetMultitaskModel

# ==========================================
# Helper Functions
# ==========================================
def get_color_map():
    """Defines the color palette for the 7 KITTI Semantic Classes (BGR format)"""
    return np.array([
        [128, 64, 128],   # 0: road
        [142, 0, 0],      # 1: car
        [35, 142, 107],   # 2: vegetation
        [70, 70, 70],     # 3: building
        [232, 130, 70],   # 4: sky
        [232, 35, 244],   # 5: sidewalk
        [153, 153, 190],  # 6: fence
        [0, 0, 0]         # Fallback
    ], dtype=np.uint8)

def decode_yolo_dfl_multiclass(preds, img_size=(192, 640), conf_thresh=0.25):
    """Decodes DFL and multi-class classification predictions for Object Detection."""
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


# ==========================================
# Argument Parser
# ==========================================
def parse_args():
    parser = argparse.ArgumentParser(description="HydraNet Multi-Task Joint Inference (Strict Architecture Match)")
    
    parser.add_argument('--weights', type=str, 
                        default=os.path.join(PROJECT_ROOT, "checkpoints", "runs", "official", "best_multitask_model.pth"),
                        help='Path to the trained joint multi-task model weights')
    
    parser.add_argument('--data_root', type=str, 
                        default=os.path.join(PROJECT_ROOT, "data"),
                        help='Root directory containing official kitti_object, kitti_semantics, kitti_depth')
    parser.add_argument('--split', type=str, default='test', choices=['val', 'test'],
                        help='Dataset split to infer')
    
    parser.add_argument('--out_dir', type=str, 
                        default=os.path.join(PROJECT_ROOT, "outputs", "official", "multitask", "test"),
                        help='Unified output directory. Will strictly branch into exactly configured od/, ss/, and de/ structures.')
    
    parser.add_argument('--img_h', type=int, default=192, help='Target image height')
    parser.add_argument('--img_w', type=int, default=640, help='Target image width')
    parser.add_argument('--conf_thresh', type=float, default=0.25, help='OD Confidence threshold')
    parser.add_argument('--iou_thresh', type=float, default=0.45, help='OD NMS IoU threshold')
    
    return parser.parse_args()


# ==========================================
# Main Inference Loop
# ==========================================
def main():
    args = parse_args()
    print("="*65)
    print("🚀 Initiating HydraNet 3-Task Joint Inference Pipeline")
    print(f"📊 Aggregating Union Set for Split: {args.split.upper()}")
    print("="*65)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Model and Weights
    model = HydraNetMultitaskModel(num_ss_classes=7, num_od_classes=7).to(device)
    if not os.path.exists(args.weights):
        raise FileNotFoundError(f"Model weights not found at {args.weights}. Train the joint model first!")
        
    checkpoint = torch.load(args.weights, map_location=device)
    model.load_state_dict(checkpoint.get('model_state_dict', checkpoint), strict=False)
    model.eval()
    print(f"✅ Joint Model loaded successfully from {args.weights}")

    # ==========================================
    # 2. Strict Output Folder Architecture Setup
    # ==========================================
    print(f"📂 Generating synchronized STRICT output structures at: {args.out_dir}")
    
    # SS Structure: images | semantic | semantic_rgb
    ss_img_dir = os.path.join(args.out_dir, "ss", "images")
    ss_semantic_dir = os.path.join(args.out_dir, "ss", "semantic")
    ss_rgb_dir = os.path.join(args.out_dir, "ss", "semantic_rgb")
    os.makedirs(ss_img_dir, exist_ok=True)
    os.makedirs(ss_semantic_dir, exist_ok=True)
    os.makedirs(ss_rgb_dir, exist_ok=True)
    
    # DE Structure: depth | images | predicted_depth
    de_depth_dir = os.path.join(args.out_dir, "de", "depth")
    de_img_dir = os.path.join(args.out_dir, "de", "images")
    de_pred_dir = os.path.join(args.out_dir, "de", "predicted_depth")
    os.makedirs(de_depth_dir, exist_ok=True)
    os.makedirs(de_img_dir, exist_ok=True)
    os.makedirs(de_pred_dir, exist_ok=True)
    
    # OD Structure: annotated_images | images | labels
    od_anno_dir = os.path.join(args.out_dir, "od", "annotated_images")
    od_img_dir = os.path.join(args.out_dir, "od", "images")
    od_txt_dir = os.path.join(args.out_dir, "od", "labels")
    os.makedirs(od_anno_dir, exist_ok=True)
    os.makedirs(od_img_dir, exist_ok=True)
    os.makedirs(od_txt_dir, exist_ok=True)

    color_map = get_color_map()
    
    # 3. Dynamic Dataset Aggregation
    task_dirs = [
        os.path.join(args.data_root, "kitti_object", args.split, "images"),
        os.path.join(args.data_root, "kitti_semantics", args.split, "images"),
        os.path.join(args.data_root, "kitti_depth", args.split, "images")
    ]
    
    image_registry = {} 
    
    for d in task_dirs:
        if os.path.exists(d):
            for f in os.listdir(d):
                if f.endswith(('.png', '.jpg')) and f not in image_registry:
                    image_registry[f] = os.path.join(d, f)
                    
    if not image_registry:
        print(f"❌ Error: No images found across any of the task directories in {args.data_root}.")
        return

    print(f"\n🔍 Found {len(image_registry)} unique images across OD, SS, and DE datasets.")
    print("Running Multi-Task Forward Pass...")
    
    with torch.no_grad():
        for i, (img_name, img_path) in enumerate(image_registry.items()):
            base_name = os.path.splitext(img_name)[0]
            
            # Read & Archive Original Image into ALL 3 strictly defined image folders
            orig_img = cv2.imread(img_path)
            orig_h, orig_w = orig_img.shape[:2]
            
            shutil.copy(img_path, os.path.join(ss_img_dir, img_name))
            shutil.copy(img_path, os.path.join(de_img_dir, img_name))
            shutil.copy(img_path, os.path.join(od_img_dir, img_name))
            
            # DE GT Archiving Setup (Copy Ground Truth depth if it exists for this image)
            gt_depth_path = os.path.join(args.data_root, "kitti_depth", args.split, "depth", img_name)
            if os.path.exists(gt_depth_path):
                shutil.copy(gt_depth_path, os.path.join(de_depth_dir, img_name))

            # Preprocess
            resized_img = cv2.resize(orig_img, (args.img_w, args.img_h), interpolation=cv2.INTER_LINEAR)
            rgb_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
            img_tensor = torch.from_numpy(rgb_img).float().permute(2, 0, 1).unsqueeze(0) / 255.0
            img_tensor = img_tensor.to(device)

            # ==========================================
            # 🚀 FORWARD PASS: ONE FOR ALL
            # ==========================================
            out_od, out_ss, out_de = model(img_tensor)

            # ==========================================
            # 🧩 Branch 1: Semantic Segmentation (SS)
            # ==========================================
            pred_mask_raw = torch.argmax(out_ss, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
            pred_mask_raw = np.clip(pred_mask_raw, 0, len(color_map) - 1)
            pred_mask_resized = cv2.resize(pred_mask_raw, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
            
            cv2.imwrite(os.path.join(ss_semantic_dir, f"{base_name}.png"), pred_mask_resized)
            cv2.imwrite(os.path.join(ss_rgb_dir, f"{base_name}.png"), color_map[pred_mask_resized])

            # ==========================================
            # 🌊 Branch 2: Depth Estimation (DE)
            # ==========================================
            pred_depth = out_de.squeeze().cpu().numpy()
            pred_depth_resized = cv2.resize(pred_depth, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            pred_depth_uint16 = (pred_depth_resized * 256.0).astype(np.uint16)
            
            cv2.imwrite(os.path.join(de_pred_dir, img_name), pred_depth_uint16)

            # ==========================================
            # 🎯 Branch 3: Object Detection (OD)
            # ==========================================
            od_render_img = orig_img.copy()
            boxes, scores, class_ids = decode_yolo_dfl_multiclass(out_od, img_size=(args.img_h, args.img_w), conf_thresh=args.conf_thresh)
            
            txt_content = ""
            if len(boxes) > 0:
                keep = batched_nms(boxes, scores, class_ids, iou_threshold=args.iou_thresh)
                for idx in keep:
                    box = boxes[idx].cpu().numpy()
                    cls_id = int(class_ids[idx].item())
                    score = scores[idx].item()
                    
                    # Rescale boxes back to original image dimensions
                    x1 = int(box[0] / args.img_w * orig_w)
                    y1 = int(box[1] / args.img_h * orig_h)
                    x2 = int(box[2] / args.img_w * orig_w)
                    y2 = int(box[3] / args.img_h * orig_h)
                    
                    # Draw Bounding Box & Label
                    cv2.rectangle(od_render_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"Class {cls_id}: {score:.2f}"
                    cv2.putText(od_render_img, label, (x1, max(y1 - 5, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    
                    # Build YOLO TXT format
                    cx, cy = (x1 + x2) / 2.0 / orig_w, (y1 + y2) / 2.0 / orig_h
                    bw, bh = (x2 - x1) / orig_w, (y2 - y1) / orig_h
                    txt_content += f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n"

            # Save OD Annotated Image and TXT Predictions to their specific strict folders
            cv2.imwrite(os.path.join(od_anno_dir, img_name), od_render_img)
            with open(os.path.join(od_txt_dir, f"{base_name}.txt"), 'w') as f:
                f.write(txt_content)

            sys.stdout.write(f"\r  👉 Processed {i+1}/{len(image_registry)} images")
            sys.stdout.flush()

    print("\n\n" + "="*65)
    print("🎉 Multi-Task Inference Complete!")
    print(f"  📂 OD Data (annotated_images/images/labels) : {os.path.join(args.out_dir, 'od')}")
    print(f"  📂 SS Data (images/semantic/semantic_rgb)   : {os.path.join(args.out_dir, 'ss')}")
    print(f"  📂 DE Data (depth/images/predicted_depth)   : {os.path.join(args.out_dir, 'de')}")
    print("="*65)

if __name__ == "__main__":
    main()