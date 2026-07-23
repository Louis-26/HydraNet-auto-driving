import os
import sys
import torch
import numpy as np
from torch.utils.data import DataLoader

# ==========================================
# 🚨 Bulletproof Path Resolution
# ==========================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dataloaders.kitti_dataset import KittiDepthDataset
from models.model import HydraNetDepthModel

def compute_depth_metrics(preds, targets):
    """
    Compute standard depth estimation metrics: RMSE, MAE, and Delta Accuracy.
    Prevents division by zero or log of zero by masking out zero-depth GT values
    and bounding predictions to positive numbers.
    """
    preds = preds.flatten()
    targets = targets.flatten()

    # 1. Filter out missing/invalid ground truth pixels (e.g., depth <= 0)
    valid_mask = targets > 1e-3
    preds = preds[valid_mask]
    targets = targets[valid_mask]
    
    # 2. 🚨 FIX: Prevent divide-by-zero when calculating thresholds
    # Deep learning models might predict 0.0 or negative values before fully converging.
    # We clip predictions to a tiny positive number to ensure mathematically safe division.
    preds = np.clip(preds, a_min=1e-5, a_max=None)
    
    # Absolute Relative Error (AbsRel)
    abs_rel = np.mean(np.abs(preds - targets) / targets)

    # Root Mean Squared Error (RMSE)
    rmse = np.sqrt(np.mean((preds - targets) ** 2))
    
    # Mean Absolute Error (MAE)
    mae = np.mean(np.abs(preds - targets))

    # Delta Accuracy (Threshold metrics: δ < 1.25, 1.25^2, 1.25^3)
    thresh = np.maximum((targets / preds), (preds / targets))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    return abs_rel, rmse, mae, a1, a2, a3

def main():
    print("="*50)
    print("🚀 Initiating Depth Estimation Evaluation")
    print("="*50)

    # 1. Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Load Model and Best Checkpoint
    model = HydraNetDepthModel().to(device)
    weight_path = os.path.join(PROJECT_ROOT, "checkpoints", "runs", "best_depth_model.pth")
    
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"Model weights not found at {weight_path}. Run training first!")
        
    checkpoint = torch.load(weight_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"✅ Successfully loaded weights from epoch {checkpoint.get('epoch', 'Unknown')} (Loss: {checkpoint.get('loss', 0.0):.4f})")

    # 3. Setup Validation/Test Dataloader 
    # (Pointing to the 'labels' structure as unified across all tasks)
    val_img_dir = os.path.join(PROJECT_ROOT, "dummy_data", "de", "test", "images")
    val_depth_dir = os.path.join(PROJECT_ROOT, "dummy_data", "de", "test", "labels")
    
    val_dataset = KittiDepthDataset(
        image_dir=val_img_dir, 
        depth_dir=val_depth_dir, 
        target_size=(192, 640)
    )
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)

    # 4. Evaluation Loop
    all_preds = []
    all_targets = []

    print("\nRunning inference across evaluation dataset...")
    with torch.no_grad():
        for images, depths in val_loader:
            images = images.to(device)
            # Forward pass: [B, 1, H, W]
            preds = model(images)
            
            all_preds.append(preds.cpu().numpy())
            all_targets.append(depths.numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # 5. Compute Metrics Globally
    abs_rel, rmse, mae, a1, a2, a3 = compute_depth_metrics(all_preds, all_targets)

    # 6. Print Detailed Report
    print("\n" + "="*20 + " Depth Evaluation Report " + "="*20)
    print(f"  - Absolute Relative Error (AbsRel) : {abs_rel:.4f}  ↓ (Lower is better)")
    print(f"  - Root Mean Squared Error (RMSE)   : {rmse:.4f}  ↓")
    print(f"  - Mean Absolute Error (MAE)        : {mae:.4f}  ↓")
    print(f"  - Accuracy δ < 1.25                : {a1*100:.2f}% ↑ (Higher is better)")
    print(f"  - Accuracy δ < 1.25²               : {a2*100:.2f}% ↑")
    print(f"  - Accuracy δ < 1.25³               : {a3*100:.2f}% ↑")
    print("="*65)
    print("🎉 Depth Estimation Evaluation Finished Successfully!")

if __name__ == "__main__":
    main()