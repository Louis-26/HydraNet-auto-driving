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

from dataloaders.kitti_dataset import KittiSegmentationDataset
from models.model import HydraNetSegmentationModel

def compute_global_iou(all_preds, all_targets, num_classes=7, ignore_index=255):
    """
    Compute dataset-level global Intersection over Union (IoU).
    Accumulates intersections and unions across all images first to avoid per-image bias.
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
            
            intersection = (pred_inds & target_inds).sum()
            union = (pred_inds | target_inds).sum()
            
            total_intersections[cls] += intersection
            total_unions[cls] += union
            
    class_ious = []
    for cls in range(num_classes):
        if total_unions[cls] == 0:
            class_ious.append(float('nan'))  # Class never appeared in the dataset
        else:
            class_ious.append(total_intersections[cls] / total_unions[cls])
            
    return class_ious

def main():
    print("="*50)
    print("🚀 Initiating Semantic Segmentation Evaluation")
    print("="*50)

    # 1. Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 7
    class_names = ['road', 'car', 'vegetation', 'building', 'sky', 'sidewalk', 'fence']

    # 2. Load Model and Best Checkpoint Weights
    model = HydraNetSegmentationModel(num_classes=num_classes).to(device)
    weight_path = os.path.join(PROJECT_ROOT, "checkpoints", "runs", "best_segmentation_model.pth")
    
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"Model weights not found at {weight_path}. Run training first!")
        
    checkpoint = torch.load(weight_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"✅ Successfully loaded weights from epoch {checkpoint.get('epoch', 'Unknown')} (Loss: {checkpoint.get('loss', 0.0):.4f})")

    # 3. Setup Validation/Test Dataloader
    val_img_dir = os.path.join(PROJECT_ROOT, "dummy_data", "ss", "test", "images")
    # 🔥 严格对齐新数据集结构：Ground Truth 标签现在从 'semantic' 文件夹读取
    val_mask_dir = os.path.join(PROJECT_ROOT, "dummy_data", "ss", "test", "semantic")
    
    val_dataset = KittiSegmentationDataset(
        image_dir=val_img_dir, 
        mask_dir=val_mask_dir, 
        target_size=(192, 640)
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=2, 
        shuffle=False, 
        drop_last=False
    )

    # 4. Inspect Ground Truth Labels Explicitly
    print("\n🔍 Inspecting Ground Truth Semantic Labels...")
    gt_classes_present = set()
    all_targets_list = []
    
    for _, masks in val_loader:
        masks_np = masks.numpy()
        all_targets_list.append(masks_np)
        # Exclude ignore_index (255) when scanning valid ground truth classes
        valid_pixels = masks_np[masks_np != 255]
        gt_classes_present.update(np.unique(valid_pixels))

    sorted_gt_classes = sorted(list(gt_classes_present))
    print(f"  - Unique class IDs found in Ground Truth: {sorted_gt_classes}")
    print(f"  - Total number of unique classes present in GT: {len(sorted_gt_classes)}")

    # 5. Evaluation Loop
    all_preds = []
    all_targets = np.concatenate(all_targets_list, axis=0)

    print("\nRunning inference across evaluation dataset...")
    with torch.no_grad():
        for images, _ in val_loader:
            images = images.to(device)
            preds = model(images)
            pred_classes = torch.argmax(preds, dim=1)
            all_preds.append(pred_classes.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)

    # 6. Compute Metrics via Global Accumulation
    class_ious = compute_global_iou(all_preds, all_targets, num_classes=num_classes)
    
    valid_ious = [iou for iou in class_ious if not np.isnan(iou)]
    mIoU = np.mean(valid_ious) * 100.0 if valid_ious else 0.0

    # 7. Print Detailed Report
    print("\n" + "="*20 + " Evaluation Report " + "="*20)
    for idx, name in enumerate(class_names):
        if idx < len(class_ious) and not np.isnan(class_ious[idx]):
            iou_val = class_ious[idx] * 100.0
            print(f"  - Class {idx} ({name:<10}): IoU = {iou_val:.2f}%")
        else:
            print(f"  - Class {idx} ({name:<10}): IoU = N/A (Absent in GT)")
    print("-" * 59)
    print(f"🏆 Mean Intersection over Union (mIoU): {mIoU:.2f}%")
    print("="*59)
    print("🎉 Semantic Segmentation Evaluation Finished Successfully!")

if __name__ == "__main__":
    main()