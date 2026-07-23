import os
import sys
import cv2
import torch
import shutil
import numpy as np

# ==========================================
# 🚨 Bulletproof Path Resolution
# ==========================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.model import HydraNetDepthModel

def main():
    print("="*50)
    print("🚀 Initiating Depth Estimation Inference (Strict PNG Format)")
    print("="*50)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load Model
    model = HydraNetDepthModel().to(device)
    weight_path = os.path.join(PROJECT_ROOT, "checkpoints", "runs", "best_depth_model.pth")
    
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"Weight not found at {weight_path}. Train the model first!")
        
    checkpoint = torch.load(weight_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"✅ Model weights loaded successfully from {weight_path}")

    # 2. Setup Correct I/O Paths
    input_img_dir = os.path.join(PROJECT_ROOT, "dummy_data", "de", "test", "images")
    input_gt_dir = os.path.join(PROJECT_ROOT, "dummy_data", "de", "test", "labels")
    
    # Output Structure: images, depth, predicted_depth
    output_base_dir = os.path.join(PROJECT_ROOT, "outputs", "dummy", "de", "test")
    out_images_dir = os.path.join(output_base_dir, "images")
    out_depth_dir = os.path.join(output_base_dir, "depth")
    out_pred_dir = os.path.join(output_base_dir, "predicted_depth")
    
    os.makedirs(out_images_dir, exist_ok=True)
    os.makedirs(out_depth_dir, exist_ok=True)
    os.makedirs(out_pred_dir, exist_ok=True)

    # 3. Inference Loop
    print("\nRunning Inference and saving 16-bit PNG depth maps...")
    image_files = [f for f in os.listdir(input_img_dir) if f.endswith(('.png', '.jpg'))]

    with torch.no_grad():
        for img_name in image_files:
            img_path = os.path.join(input_img_dir, img_name)
            gt_path = os.path.join(input_gt_dir, img_name)
            
            # --- Step A: Archive Original Inputs ---
            shutil.copy(img_path, os.path.join(out_images_dir, img_name))
            if os.path.exists(gt_path):
                shutil.copy(gt_path, os.path.join(out_depth_dir, img_name))
            
            # --- Step B: Preprocessing ---
            orig_img = cv2.imread(img_path)
            orig_h, orig_w = orig_img.shape[:2]
            
            resized_img = cv2.resize(orig_img, (640, 192), interpolation=cv2.INTER_LINEAR)
            rgb_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
            
            img_tensor = torch.from_numpy(rgb_img).float().permute(2, 0, 1).unsqueeze(0) / 255.0
            img_tensor = img_tensor.to(device)

            # --- Step C: Forward Pass ---
            preds = model(img_tensor) # Shape: [1, 1, 192, 640]
            pred_depth = preds.squeeze().cpu().numpy() 

            # --- Step D: Postprocessing & Save as 16-bit PNG ---
            # 1. Resize back to original dimensions
            pred_depth_resized = cv2.resize(pred_depth, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            
            # 2. Convert to KITTI 16-bit format (multiplying by 256.0)
            pred_depth_uint16 = (pred_depth_resized * 256.0).astype(np.uint16)
            
            # 3. Save purely as .png (same name as input)
            pred_save_path = os.path.join(out_pred_dir, img_name)
            cv2.imwrite(pred_save_path, pred_depth_uint16)
            
            print(f"  - Saved identical format prediction: {img_name}")

    print("====================================================")
    print(f"🎉 Depth Inference Complete! Check your strictly formatted outputs at: {output_base_dir}")

if __name__ == "__main__":
    main()