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

from models.model import HydraNetDetectionModel

# Class Mapping for Visualization
ID_TO_CLASS = {
    0: "Car", 1: "Van", 2: "Truck", 3: "Pedestrian",
    4: "Person_sitting", 5: "Cyclist", 6: "Tram"
}

# Distinct colors for each class (BGR format)
CLASS_COLORS = {
    0: (0, 255, 0),    # Green
    1: (255, 255, 0),  # Cyan
    2: (0, 165, 255),  # Orange
    3: (0, 0, 255),    # Red
    4: (255, 0, 255),  # Magenta
    5: (255, 0, 0),    # Blue
    6: (128, 128, 128) # Gray
}

def parse_args():
    parser = argparse.ArgumentParser(description="HydraNet Object Detection Inference")
    parser.add_argument('--weights', type=str, default='checkpoints/runs/best_detection_model.pth',
                        help='Path to the trained model weights')
    parser.add_argument('--source', type=str, default='dummy_data/od/test/images',
                        help='Directory containing images to infer')
    parser.add_argument('--out_dir', type=str, default='outputs/dummy/od/test',
                        help='Output directory for visuals and labels')
    parser.add_argument('--conf_thresh', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--iou_thresh', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--num_classes', type=int, default=7, help='Number of classes')
    parser.add_argument('--img_h', type=int, default=192, help='Target image height')
    parser.add_argument('--img_w', type=int, default=640, help='Target image width')
    return parser.parse_args()

def preprocess_image(img_path, target_size=(192, 640)):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Could not read image at {img_path}")
    
    orig_img = img.copy() 
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (target_size[1], target_size[0]))
    
    img_tensor = torch.from_numpy(img).float() / 255.0
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0) 
    return orig_img, img_tensor

def decode_yolo_dfl(preds, img_size=(192, 640), conf_thresh=0.25):
    """
    Decodes DFL and multi-class classification predictions.
    """
    all_bboxes, all_scores, all_class_ids = [], [], []
    reg_max = 16 
    dfl_weights = torch.arange(reg_max, dtype=torch.float32, device=preds[0]['bbox'].device)
    img_h, img_w = img_size

    for pred in preds:
        bbox_feat, cls_feat = pred['bbox'], pred['cls']
        B, C, grid_h, grid_w = cls_feat.shape
        stride_h, stride_w = img_h / grid_h, img_w / grid_w
        
        # 1. Multi-class classification logic
        cls_scores = torch.sigmoid(cls_feat).squeeze(0)
        max_scores, max_class_ids = torch.max(cls_scores, dim=0)
        
        # 2. Bounding Box DFL logic
        bbox_feat = pred['bbox'].view(4, reg_max, grid_h, grid_w).permute(2, 3, 0, 1)
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
        all_bboxes.append(bboxes[mask])
        all_scores.append(scores[mask])
        all_class_ids.append(class_ids[mask])
        
    return torch.cat(all_bboxes), torch.cat(all_scores), torch.cat(all_class_ids)

def main():
    args = parse_args()
    print("=============================================")
    print("🚀 Initiating Inference Pipeline...")
    print(f"Source: {args.source}")
    print(f"Weights: {args.weights}")
    print("=============================================")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Initialize Model and Load Weights
    model = HydraNetDetectionModel(num_classes=args.num_classes).to(device)
    weight_path = os.path.join(PROJECT_ROOT, args.weights)
    
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"Weights not found at {weight_path}. Did you train the model?")
        
    checkpoint = torch.load(weight_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval() 
    print("✅ Model weights loaded successfully.")

    # 2. Prepare Output Directories
    img_out_dir = os.path.join(PROJECT_ROOT, args.out_dir, "images")
    vis_dir = os.path.join(PROJECT_ROOT, args.out_dir, "annotated_images")
    label_dir = os.path.join(PROJECT_ROOT, args.out_dir, "labels")
    
    os.makedirs(img_out_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)

    # 3. Iterate through Source Images
    source_dir = os.path.join(PROJECT_ROOT, args.source)
    img_files = [f for f in os.listdir(source_dir) if f.endswith(('.png', '.jpg'))]
    
    if not img_files:
        print(f"No images found in {source_dir}.")
        return

    for img_name in img_files:
        test_img_path = os.path.join(source_dir, img_name)
        
        # Archive original raw image
        shutil.copy(test_img_path, os.path.join(img_out_dir, img_name))

        orig_img, img_tensor = preprocess_image(test_img_path, target_size=(args.img_h, args.img_w))
        img_tensor = img_tensor.to(device)

        with torch.no_grad():
            preds = model(img_tensor)

        # 4. Decode & Batched NMS
        bboxes, scores, class_ids = decode_yolo_dfl(preds, img_size=(args.img_h, args.img_w), conf_thresh=args.conf_thresh)
        
        txt_path = os.path.join(label_dir, os.path.splitext(img_name)[0] + ".txt")
        
        if len(bboxes) == 0:
            print(f"[{img_name}] No detections found.")
            open(txt_path, 'w').close()
            cv2.imwrite(os.path.join(vis_dir, img_name), orig_img)
            continue
            
        keep_indices = torchvision.ops.batched_nms(bboxes, scores, class_ids, iou_threshold=args.iou_thresh)
        
        final_bboxes = bboxes[keep_indices]
        final_scores = scores[keep_indices]
        final_classes = class_ids[keep_indices]

        # 5. Draw Visuals & Save Labels with Prominent Label Names
        orig_h, orig_w = orig_img.shape[:2]
        scale_w = orig_w / args.img_w
        scale_h = orig_h / args.img_h

        with open(txt_path, "w") as f:
            for i in range(len(final_bboxes)):
                x1, y1, x2, y2 = final_bboxes[i].cpu().numpy()
                score = final_scores[i].item()
                cls_id = int(final_classes[i].item())
                cls_name = ID_TO_CLASS.get(cls_id, "Unknown")
                color = CLASS_COLORS.get(cls_id, (255, 255, 255))
                
                x1 = np.clip(x1, 0, args.img_w)
                y1 = np.clip(y1, 0, args.img_h)
                x2 = np.clip(x2, 0, args.img_w)
                y2 = np.clip(y2, 0, args.img_h)
                
                # YOLO Prediction Format Export (Strictly bounded between 0.0 and 1.0)
                x_center = np.clip(((x1 + x2) / 2) / args.img_w, 0.0, 1.0)
                y_center = np.clip(((y1 + y2) / 2) / args.img_h, 0.0, 1.0)
                w_norm = np.clip((x2 - x1) / args.img_w, 0.0, 1.0)
                h_norm = np.clip((y2 - y1) / args.img_h, 0.0, 1.0)
                
                # Write safe 5-column format
                f.write(f"{cls_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")

                # Rescale coordinates to original image resolution for visualization
                real_x1, real_x2 = int(x1 * scale_w), int(x2 * scale_w)
                real_y1, real_y2 = int(y1 * scale_h), int(y2 * scale_h)
                
                cv2.rectangle(orig_img, (real_x1, real_y1), (real_x2, real_y2), color, 2)
                
                # 🌟 Prominent Label Name & Score Rendering with boundary check
                label_text = f"{cls_name} ({score:.2f})"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                thickness = 1
                
                (txt_w, txt_h), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)
                
                # If box is too close to the top edge, render label inside/below the box top edge
                text_y = real_y1 - 4 if real_y1 - txt_h - 4 > 0 else real_y1 + txt_h + 4
                
                # Draw filled background rectangle for absolute clarity of the label name
                cv2.rectangle(orig_img, (real_x1, text_y - txt_h - baseline), (real_x1 + txt_w, text_y + baseline), color, -1)
                cv2.putText(orig_img, label_text, (real_x1, text_y - baseline), font, font_scale, (0, 0, 0), thickness)

        cv2.imwrite(os.path.join(vis_dir, img_name), orig_img)
        print(f"[{img_name}] Found {len(final_bboxes)} objects.")

    print(f"\n✅ All done! Results saved cleanly in: {os.path.join(PROJECT_ROOT, args.out_dir)}")

if __name__ == "__main__":
    main()