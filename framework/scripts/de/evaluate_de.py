import os
import sys
import argparse
import cv2
import numpy as np

# ==========================================
# Bulletproof Path Resolution
# ==========================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def parse_args():
    parser = argparse.ArgumentParser(description="HydraNet Offline Depth Estimation Evaluation")
    
    # Path configurations for saved predictions and dataset ground truth
    parser.add_argument('--pred_dir', type=str, 
                        default=os.path.join(PROJECT_ROOT, "outputs", "official", "de", "predicted_depth"),
                        help='Directory containing saved 16-bit predicted depth maps (PNGs)')
    parser.add_argument('--data_root', type=str, 
                        default=os.path.join(PROJECT_ROOT, "data", "kitti_depth"),
                        help='Root directory of depth estimation dataset')
    parser.add_argument('--split', type=str, default='test', choices=['val', 'test', 'train'],
                        help='Dataset split being evaluated')
    
    return parser.parse_args()


def compute_depth_metrics(preds, targets):
    """
    Compute standard depth estimation metrics: RMSE, MAE, and Delta Accuracy.
    Expects flattened 1D arrays of valid pixels only.
    """
    # 🚨 FIX: Prevent divide-by-zero when calculating thresholds
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
    args = parse_args()
    print("="*60)
    print("🚀 Initiating Offline Depth Estimation Evaluation")
    print(f"📂 Prediction Directory : {args.pred_dir}")
    print(f"📊 Dataset Split        : {args.split.upper()}")
    print("="*60)

    # Setup Ground Truth Directory
    gt_depth_dir = os.path.join(args.data_root, args.split, "depth")
    
    if not os.path.exists(args.pred_dir):
        raise FileNotFoundError(f"Prediction directory not found at {args.pred_dir}. Run inference first!")
    if not os.path.exists(gt_depth_dir):
        raise FileNotFoundError(f"Ground truth depth directory not found at {gt_depth_dir}.")

    # Collect prediction files
    pred_files = [f for f in os.listdir(args.pred_dir) if f.endswith(('.png', '.jpg'))]
    if not pred_files:
        print(f"🤷‍♂️ No prediction images found in {args.pred_dir}.")
        return

    valid_preds = []
    valid_targets = []

    print(f"\nMatching and loading {len(pred_files)} prediction-ground truth pairs...")
    for file_name in pred_files:
        pred_path = os.path.join(args.pred_dir, file_name)
        gt_path = os.path.join(gt_depth_dir, file_name)

        if not os.path.exists(gt_path):
            print(f"⚠️ Warning: Ground truth for {file_name} not found in {gt_depth_dir}. Skipping.")
            continue

        # 🚨 Use cv2.IMREAD_UNCHANGED to read 16-bit PNGs without losing data
        pred_img = cv2.imread(pred_path, cv2.IMREAD_UNCHANGED)
        gt_img = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)

        if pred_img is None or gt_img is None:
            print(f"⚠️ Warning: Failed to read image or mask for {file_name}. Skipping.")
            continue

        # Convert 16-bit integer back to float depth (KITTI standard: depth = pixel / 256.0)
        pred_depth = pred_img.astype(np.float32) / 256.0
        gt_depth = gt_img.astype(np.float32) / 256.0

        # Handle size mismatches gracefully by resizing prediction
        if pred_depth.shape != gt_depth.shape:
            pred_depth = cv2.resize(pred_depth, (gt_depth.shape[1], gt_depth.shape[0]), interpolation=cv2.INTER_LINEAR)

        # Memory Optimization: Filter out missing/invalid ground truth pixels immediately
        valid_mask = gt_depth > 1e-3
        
        if valid_mask.sum() > 0:
            valid_preds.append(pred_depth[valid_mask])
            valid_targets.append(gt_depth[valid_mask])

    if len(valid_preds) == 0:
        print("❌ Error: No valid evaluation pairs matched!")
        return

    print("Concatenating pixels and computing metrics globally...")
    # Concatenate all valid pixels into massive 1D arrays
    all_preds = np.concatenate(valid_preds, axis=0)
    all_targets = np.concatenate(valid_targets, axis=0)

    # Compute Metrics Globally
    abs_rel, rmse, mae, a1, a2, a3 = compute_depth_metrics(all_preds, all_targets)

    # Print Detailed Report
    print("\n" + "="*20 + " Offline Depth Evaluation Report " + "="*20)
    print(f"🎯 Evaluated Samples                 : {len(pred_files)} images")
    print(f"  - Absolute Relative Error (AbsRel) : {abs_rel:.4f}  ↓ (Lower is better)")
    print(f"  - Root Mean Squared Error (RMSE)   : {rmse:.4f}  ↓")
    print(f"  - Mean Absolute Error (MAE)        : {mae:.4f}  ↓")
    print(f"  - Accuracy δ < 1.25                : {a1*100:.2f}% ↑ (Higher is better)")
    print(f"  - Accuracy δ < 1.25²               : {a2*100:.2f}% ↑")
    print(f"  - Accuracy δ < 1.25³               : {a3*100:.2f}% ↑")
    print("="*71)
    print("🎉 Offline Depth Estimation Evaluation Finished Successfully!")


if __name__ == "__main__":
    main()