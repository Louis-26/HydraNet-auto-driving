import os
import sys
import shutil
import argparse
import cv2
import torch
import numpy as np

# ==========================================
# Bulletproof Path Resolution
# ==========================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.model import HydraNetSegmentationModel

def get_color_map():
    """
    Defines the color palette for the 7 KITTI classes.
    Format: BGR (for OpenCV compatibility)
    """
    return np.array([
        [128, 64, 128],   # 0: road (Purple)
        [142, 0, 0],      # 1: car (Dark Blue)
        [35, 142, 107],   # 2: vegetation (Green)
        [70, 70, 70],     # 3: building (Gray)
        [232, 130, 70],   # 4: sky (Light Blue)
        [232, 35, 244],   # 5: sidewalk (Pink)
        [153, 153, 190],  # 6: fence (Beige/Orange)
        [0, 0, 0]         # Fallback for ignored/unknown (Black)
    ], dtype=np.uint8)

def parse_args():
    parser = argparse.ArgumentParser(description="HydraNet Semantic Segmentation Inference")
    
    parser.add_argument('--weights', type=str, 
                        default=os.path.join(PROJECT_ROOT, "checkpoints", "runs", "official", "best_segmentation_model.pth"),
                        help='Path to official segmentation model weights')
    parser.add_argument('--source', type=str, 
                        default=os.path.join(PROJECT_ROOT, "data", "kitti_semantics", "test", "images"),
                        help='Directory containing test images to infer')
    parser.add_argument('--out_dir', type=str, 
                        default=os.path.join(PROJECT_ROOT, "outputs", "official", "ss", "test"),
                        help='Output directory for standardized predictions')
    parser.add_argument('--num_classes', type=int, default=7, help='Number of segmentation classes')
    parser.add_argument('--img_h', type=int, default=192, help='Target image height')
    parser.add_argument('--img_w', type=int, default=640, help='Target image width')
    
    return parser.parse_args()

def main():
    args = parse_args()
    print("="*60)
    print("🚀 Initiating Semantic Segmentation Inference Pipeline")
    print("="*60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load the Model
    model = HydraNetSegmentationModel(num_classes=args.num_classes).to(device)
    
    if not os.path.exists(args.weights):
        raise FileNotFoundError(f"Weight not found at {args.weights}. Train the model first!")
        
    checkpoint = torch.load(args.weights, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"✅ Model weights loaded successfully from {args.weights}")

    # 2. Setup Correct I/O Paths (Strictly aligned with 3 dataset folders)
    input_dir = args.source
    
    out_images_dir = os.path.join(args.out_dir, "images")
    out_semantic_dir = os.path.join(args.out_dir, "semantic")
    out_semantic_rgb_dir = os.path.join(args.out_dir, "semantic_rgb")
    
    os.makedirs(out_images_dir, exist_ok=True)
    os.makedirs(out_semantic_dir, exist_ok=True)
    os.makedirs(out_semantic_rgb_dir, exist_ok=True)

    color_map = get_color_map()

    # 3. Inference Loop
    print(f"\nRunning inference on images from: {input_dir}")
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input source directory not found: {input_dir}")
        
    image_files = [f for f in os.listdir(input_dir) if f.endswith(('.png', '.jpg'))]
    if not image_files:
        print(f"🤷‍♂️ No images found in {input_dir}.")
        return

    with torch.no_grad():
        for img_name in image_files:
            img_path = os.path.join(input_dir, img_name)
            base_name = os.path.splitext(img_name)[0]
            
            # --- 1. Copy original image to images folder ---
            shutil.copy(img_path, os.path.join(out_images_dir, img_name))
            
            # --- Preprocessing ---
            orig_img = cv2.imread(img_path)
            orig_h, orig_w = orig_img.shape[:2]
            
            resized_img = cv2.resize(orig_img, (args.img_w, args.img_h), interpolation=cv2.INTER_LINEAR)
            rgb_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
            
            img_tensor = torch.from_numpy(rgb_img).float().permute(2, 0, 1).unsqueeze(0) / 255.0
            img_tensor = img_tensor.to(device)

            # --- Forward Pass ---
            preds = model(img_tensor) 
            
            # Extract class ID map and convert to uint8
            pred_mask_raw = torch.argmax(preds, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
            pred_mask_raw = np.clip(pred_mask_raw, 0, len(color_map) - 1)

            # --- Bulletproof: Restore original size using nearest neighbor to protect class IDs ---
            pred_mask_resized = cv2.resize(pred_mask_raw, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

            # --- 2. Save Raw Semantic ID Map (Single-channel grayscale) ---
            cv2.imwrite(os.path.join(out_semantic_dir, f"{base_name}.png"), pred_mask_resized)
            
            # --- 3. Save Colorized Visualization (semantic_rgb) ---
            color_mask_resized = color_map[pred_mask_resized]
            cv2.imwrite(os.path.join(out_semantic_rgb_dir, f"{base_name}.png"), color_mask_resized)
            
            print(f"  - Structured outputs generated for: {img_name}")

    print("====================================================")
    print(f"🎉 SS Synchronized Inference Complete!")
    print(f"  📂 Outputs perfectly aligned with dataset structure at: {args.out_dir}")

if __name__ == "__main__":
    main()