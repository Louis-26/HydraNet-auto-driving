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
    parser = argparse.ArgumentParser(description="HydraNet Offline Semantic Segmentation Evaluation")
    parser.add_argument('--pred_dir', type=str, 
                        default=os.path.join(PROJECT_ROOT, "outputs", "official", "ss", "test", "semantic"),
                        help='Directory containing saved raw prediction masks (0~6 IDs)')
    parser.add_argument('--data_root', type=str, 
                        default=os.path.join(PROJECT_ROOT, "data", "kitti_semantics"),
                        help='Root directory of semantic segmentation dataset')
    parser.add_argument('--split', type=str, default='test', choices=['val', 'test', 'train'],
                        help='Dataset split being evaluated')
    parser.add_argument('--num_classes', type=int, default=7, help='Number of segmentation classes')
    return parser.parse_args()


def compute_global_iou(all_preds, all_targets, num_classes=7, ignore_index=255):
    """
    Compute dataset-level global Intersection over Union (IoU).
    """
    total_intersections = np.zeros(num_classes)
    total_unions = np.zeros(num_classes)
    
    for preds, targets in zip(all_preds, all_targets):
        preds = preds.flatten()
        targets = targets.flatten()
        
        # Filter out ignore_index pixels (e.g., 255)
        valid_mask = (targets != ignore_index)
        preds = preds[valid_mask]
        targets = targets[valid_mask]
        
        for cls in range(num_classes):
            pred_inds = (preds == cls)
            target_inds = (targets == cls)
            
            total_intersections[cls] += (pred_inds & target_inds).sum()
            total_unions[cls] += (pred_inds | target_inds).sum()
            
    class_ious = []
    for cls in range(num_classes):
        if total_unions[cls] == 0:
            class_ious.append(float('nan'))  # Class never appeared in the dataset
        else:
            class_ious.append(total_intersections[cls] / total_unions[cls])
            
    return class_ious


# ==========================================
# 🛡️ 核心修复区：严格对齐 Dataloader 逻辑
# ==========================================
def preprocess_ground_truth(raw_gt_mask):
    """
    Maps raw KITTI semantic classes (34 classes) to the 7 target classes
    used by HydraNet, matching the KittiMultitaskDataset exactly.
    """
    mapping = np.full(256, 255, dtype=np.uint8)
    mapping[[7, 26, 21, 11, 23, 8, 13]] = [0, 1, 2, 3, 4, 5, 6]
    
    # Apply the lookup table mapping instantly
    return mapping[raw_gt_mask]


def main():
    args = parse_args()
    print("="*60)
    print("🚀 Initiating Offline Evaluation on Inference Results")
    print(f"📂 Prediction Directory : {args.pred_dir}")
    print(f"📊 Dataset Split        : {args.split.upper()}")
    print("="*60)

    class_names = ['road', 'car', 'vegetation', 'building', 'sky', 'sidewalk', 'fence']

    # Ground truth mask folder aligned with official structure
    gt_mask_dir = os.path.join(args.data_root, args.split, "semantic")
    
    if not os.path.exists(args.pred_dir):
        raise FileNotFoundError(f"Prediction directory not found at {args.pred_dir}. Run inference first!")
    if not os.path.exists(gt_mask_dir):
        raise FileNotFoundError(f"Ground truth mask directory not found at {gt_mask_dir}.")

    # Collect prediction files
    pred_files = [f for f in os.listdir(args.pred_dir) if f.endswith(('.png', '.jpg'))]
    if not pred_files:
        print(f"🤷‍♂️ No prediction images found in {args.pred_dir}.")
        return

    all_preds = []
    all_targets = []

    print(f"\nMatching and loading {len(pred_files)} prediction-ground truth pairs...")
    for file_name in pred_files:
        pred_path = os.path.join(args.pred_dir, file_name)
        gt_path = os.path.join(gt_mask_dir, file_name)

        if not os.path.exists(gt_path):
            print(f"⚠️ Warning: Ground truth for {file_name} not found in {gt_mask_dir}. Skipping.")
            continue

        # Load prediction mask (already 0~6 because of inference_ss.py)
        pred_mask = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
        
        # Load raw Ground Truth (contains 7, 26, 21...)
        raw_gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

        if pred_mask is None or raw_gt_mask is None:
            print(f"⚠️ Warning: Failed to read image or mask for {file_name}. Skipping.")
            continue

        # 🚨 Apply the mapping to convert raw GT into 0~6 and 255
        gt_mask = preprocess_ground_truth(raw_gt_mask)

        # Handle size mismatches gracefully using Nearest Neighbor to protect IDs
        if pred_mask.shape != gt_mask.shape:
            pred_mask = cv2.resize(pred_mask, (gt_mask.shape[1], gt_mask.shape[0]), interpolation=cv2.INTER_NEAREST)

        all_preds.append(pred_mask)
        all_targets.append(gt_mask)

    if len(all_preds) == 0:
        print("❌ Error: No valid evaluation pairs matched!")
        return

    # Compute Metrics via Global Accumulation
    class_ious = compute_global_iou(all_preds, all_targets, num_classes=args.num_classes)
    
    valid_ious = [iou for iou in class_ious if not np.isnan(iou)]
    mIoU = np.mean(valid_ious) * 100.0 if valid_ious else 0.0

    # Print Detailed Report
    print("\n" + "="*20 + " Offline Evaluation Report " + "="*20)
    print(f"🎯 Evaluated Samples: {len(all_preds)} images")
    for idx, name in enumerate(class_names):
        if idx < len(class_ious) and not np.isnan(class_ious[idx]):
            iou_val = class_ious[idx] * 100.0
            print(f"  - Class {idx} ({name:<10}): IoU = {iou_val:.2f}%")
        else:
            print(f"  - Class {idx} ({name:<10}): IoU = N/A (Absent in GT)")
    print("-" * 59)
    print(f"🏆 Mean Intersection over Union (mIoU): {mIoU:.2f}%")
    print("="*59)
    print("🎉 Offline Semantic Segmentation Evaluation Finished Successfully!")


if __name__ == "__main__":
    main()