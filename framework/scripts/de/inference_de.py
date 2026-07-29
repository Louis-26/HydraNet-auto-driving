import os
import sys
import cv2
import torch
import shutil
import argparse
import numpy as np

# ==========================================
# Bulletproof Path Resolution
# ==========================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.model import HydraNetDepthModel

def parse_args():
    parser = argparse.ArgumentParser(description="HydraNet Depth Estimation Inference")
    
    parser.add_argument('--weights', type=str, 
                        default=os.path.join(PROJECT_ROOT, "checkpoints", "runs", "official", "best_depth_model.pth"),
                        help='Path to official depth estimation model weights')
    parser.add_argument('--source_img', type=str, 
                        default=os.path.join(PROJECT_ROOT, "data", "kitti_depth", "test", "images"),
                        help='Directory containing test images to infer')
    parser.add_argument('--source_gt', type=str, 
                        default=os.path.join(PROJECT_ROOT, "data", "kitti_depth", "test", "depth"),
                        help='(Optional) Directory containing ground truth depth maps to archive')
    parser.add_argument('--out_dir', type=str, 
                        default=os.path.join(PROJECT_ROOT, "outputs", "official", "de", "test"),
                        help='Output directory for standardized predictions')
    parser.add_argument('--img_h', type=int, default=192, help='Target image height')
    parser.add_argument('--img_w', type=int, default=640, help='Target image width')
    
    return parser.parse_args()


def main():
    args = parse_args()
    print("="*60)
    print("🚀 Initiating Depth Estimation Inference Pipeline (Strict 16-bit PNG)")
    print("="*60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Initialize Model and Load Weights
    model = HydraNetDepthModel().to(device)
    
    if not os.path.exists(args.weights):
        raise FileNotFoundError(f"Weight not found at {args.weights}. Train the model first!")
        
    checkpoint = torch.load(args.weights, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"✅ Model weights loaded successfully from {args.weights}")

    # 2. Setup Correct I/O Paths
    input_img_dir = args.source_img
    input_gt_dir = args.source_gt
    
    # Output Structure: images, depth (GT archive), predicted_depth
    out_images_dir = os.path.join(args.out_dir, "images")
    out_depth_dir = os.path.join(args.out_dir, "depth")
    out_pred_dir = os.path.join(args.out_dir, "predicted_depth")
    
    os.makedirs(out_images_dir, exist_ok=True)
    os.makedirs(out_depth_dir, exist_ok=True)
    os.makedirs(out_pred_dir, exist_ok=True)

    # 3. Inference Loop
    print(f"\nRunning Inference and saving 16-bit PNG depth maps to {args.out_dir}...")
    if not os.path.exists(input_img_dir):
        raise FileNotFoundError(f"Input source directory not found: {input_img_dir}")
        
    image_files = [f for f in os.listdir(input_img_dir) if f.endswith(('.png', '.jpg'))]
    if not image_files:
        print(f"🤷‍♂️ No images found in {input_img_dir}.")
        return

    with torch.no_grad():
        for img_name in image_files:
            img_path = os.path.join(input_img_dir, img_name)
            gt_path = os.path.join(input_gt_dir, img_name) if input_gt_dir else None
            
            # --- Step A: Archive Original Inputs ---
            shutil.copy(img_path, os.path.join(out_images_dir, img_name))
            if gt_path and os.path.exists(gt_path):
                shutil.copy(gt_path, os.path.join(out_depth_dir, img_name))
            
            # --- Step B: Preprocessing ---
            orig_img = cv2.imread(img_path)
            orig_h, orig_w = orig_img.shape[:2]
            
            resized_img = cv2.resize(orig_img, (args.img_w, args.img_h), interpolation=cv2.INTER_LINEAR)
            rgb_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
            
            img_tensor = torch.from_numpy(rgb_img).float().permute(2, 0, 1).unsqueeze(0) / 255.0
            img_tensor = img_tensor.to(device)

            # --- Step C: Forward Pass ---
            preds = model(img_tensor) # Expected Shape: [1, 1, 192, 640]
            pred_depth = preds.squeeze().cpu().numpy() 

            # --- Step D: Postprocessing & Save as 16-bit PNG ---
            # 1. Resize back to original dimensions for strict evaluation compatibility
            pred_depth_resized = cv2.resize(pred_depth, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            
            # 2. Convert to KITTI 16-bit format (multiplying absolute depth by 256.0)
            pred_depth_uint16 = (pred_depth_resized * 256.0).astype(np.uint16)
            
            # 3. Save strictly as .png (matching input filename)
            pred_save_path = os.path.join(out_pred_dir, img_name)
            cv2.imwrite(pred_save_path, pred_depth_uint16)
            
            print(f"  - Generated 16-bit depth map for: {img_name}")

    print("====================================================")
    print(f"🎉 Depth Inference Complete! Check your strictly formatted outputs at: {args.out_dir}")


if __name__ == "__main__":
    main()