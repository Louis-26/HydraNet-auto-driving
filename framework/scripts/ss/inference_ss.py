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

def main():
    print("====================================================")
    print("🚀 Initiating Semantic Segmentation Inference")
    print("====================================================")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 7

    # 1. Load the Model
    model = HydraNetSegmentationModel(num_classes=num_classes).to(device)
    weight_path = os.path.join(PROJECT_ROOT, "checkpoints", "runs", "best_segmentation_model.pth")
    
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"Weight not found at {weight_path}. Train the model first!")
        
    checkpoint = torch.load(weight_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"✅ Model weights loaded successfully from {weight_path}")

    # 2. Setup Correct I/O Paths
    input_dir = os.path.join(PROJECT_ROOT, "dummy_data", "ss", "test", "images")
    
    # 🔥 严格对齐数据集结构的 3 个输出文件夹
    output_base_dir = os.path.join(PROJECT_ROOT, "outputs", "dummy", "ss", "test")
    out_images_dir = os.path.join(output_base_dir, "images")
    out_semantic_dir = os.path.join(output_base_dir, "semantic")
    out_semantic_rgb_dir = os.path.join(output_base_dir, "semantic_rgb")
    
    os.makedirs(out_images_dir, exist_ok=True)
    os.makedirs(out_semantic_dir, exist_ok=True)
    os.makedirs(out_semantic_rgb_dir, exist_ok=True)

    color_map = get_color_map()

    # 3. Inference Loop
    print("\nRunning Inference and generating standardized outputs...")
    image_files = [f for f in os.listdir(input_dir) if f.endswith(('.png', '.jpg'))]

    with torch.no_grad():
        for img_name in image_files:
            img_path = os.path.join(input_dir, img_name)
            base_name = os.path.splitext(img_name)[0]
            
            # --- 1. 拷贝原图到 images ---
            shutil.copy(img_path, os.path.join(out_images_dir, img_name))
            
            # --- Preprocessing ---
            orig_img = cv2.imread(img_path)
            orig_h, orig_w = orig_img.shape[:2]
            
            resized_img = cv2.resize(orig_img, (640, 192), interpolation=cv2.INTER_LINEAR)
            rgb_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
            
            img_tensor = torch.from_numpy(rgb_img).float().permute(2, 0, 1).unsqueeze(0) / 255.0
            img_tensor = img_tensor.to(device)

            # --- Forward Pass ---
            preds = model(img_tensor) 
            
            # 提取类别 ID 图并转为 uint8
            pred_mask_raw = torch.argmax(preds, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
            pred_mask_raw = np.clip(pred_mask_raw, 0, len(color_map) - 1)

            # --- 🛡️ 极致防弹：先用最近邻插值还原尺寸，保护类别 ID 绝对不变 ---
            pred_mask_resized = cv2.resize(pred_mask_raw, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

            # --- 2. 保存 Raw Semantic ID Map (单通道灰度图) ---
            cv2.imwrite(os.path.join(out_semantic_dir, f"{base_name}.png"), pred_mask_resized)
            
            # --- 3. 保存 彩色可视化图 (semantic_rgb) ---
            color_mask_resized = color_map[pred_mask_resized]
            cv2.imwrite(os.path.join(out_semantic_rgb_dir, f"{base_name}.png"), color_mask_resized)
            
            print(f"  - Structured outputs generated for: {img_name}")

    print("====================================================")
    print(f"🎉 SS Synchronized Inference Complete!")
    print(f"  📂 Outputs perfectly aligned with dataset structure at: {output_base_dir}")

if __name__ == "__main__":
    main()