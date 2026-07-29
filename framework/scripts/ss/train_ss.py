import os
import sys
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
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

# ==========================================
# 🧠 Segmentation Loss Module (CE + Dice)
# ==========================================
class SegmentationLoss(nn.Module):
    def __init__(self, num_classes, dice_weight=1.0, eps=1e-5):
        super(SegmentationLoss, self).__init__()
        self.ce = nn.CrossEntropyLoss(ignore_index=255) 
        self.dice_weight = dice_weight
        self.eps = eps
        self.num_classes = num_classes

    def forward(self, preds, targets):
        ce_loss = self.ce(preds, targets)
        
        preds_soft = torch.softmax(preds, dim=1) 
        valid_mask = (targets != 255).unsqueeze(1).float()
        targets_safe = torch.where(targets == 255, torch.zeros_like(targets), targets)
        targets_one_hot = F.one_hot(targets_safe, num_classes=self.num_classes).permute(0, 3, 1, 2).float()

        intersection = torch.sum(preds_soft * targets_one_hot * valid_mask, dim=(2, 3))
        union = torch.sum(preds_soft * valid_mask, dim=(2, 3)) + torch.sum(targets_one_hot * valid_mask, dim=(2, 3))
        
        dice_loss = 1.0 - (2.0 * intersection + self.eps) / (union + self.eps)
        dice_loss = dice_loss.mean()

        return ce_loss + self.dice_weight * dice_loss


# ==========================================
# Argument Parser
# ==========================================
def parse_args():
    parser = argparse.ArgumentParser(description="HydraNet Semantic Segmentation Training")

    # Dataset & Dimensions
    parser.add_argument("--data_root", type=str, default=os.path.join(PROJECT_ROOT, "dummy_data", "ss"), help="Root directory of segmentation dataset")
    parser.add_argument("--num_classes", type=int, default=7, help="Number of segmentation classes")
    parser.add_argument("--img_h", type=int, default=192, help="Input image height")
    parser.add_argument("--img_w", type=int, default=640, help="Input image width")

    # Training Hyperparameters
    parser.add_argument("--epochs", type=int, default=50, help="Total training epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size per GPU")
    parser.add_argument("--lr", type=float, default=1e-4, help="Initial learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay for optimizer")
    parser.add_argument("--dice_weight", type=float, default=1.0, help="Weight for Dice Loss component")

    # Runtime & Checkpoint
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Computation device")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of dataloader workers")
    parser.add_argument("--pin_memory", action="store_true", help="Pin memory for faster data transfer")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--save_dir", default=os.path.join(PROJECT_ROOT, "checkpoints", "runs", "official"), help="Directory to save weights")
    parser.add_argument("--print_freq", type=int, default=5, help="Batch printing frequency")

    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


# ==========================================
# 🚀 Main Training Loop
# ==========================================
def main():
    args = parse_args()
    set_seed(args.seed)

    print("="*55)
    print("🚀 Initiating Semantic Segmentation Training Pipeline")
    print("="*55)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Dataset Paths (Aligned with 'images' and 'semantic' structure) ---
    train_img_dir = os.path.join(args.data_root, "train", "images")
    train_mask_dir = os.path.join(args.data_root, "train", "semantic")
    os.makedirs(args.save_dir, exist_ok=True)

    # --- Dataloader Setup ---
    train_dataset = KittiSegmentationDataset(image_dir=train_img_dir, mask_dir=train_mask_dir, target_size=(args.img_h, args.img_w))
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, 
        num_workers=args.num_workers, pin_memory=args.pin_memory, drop_last=False
    )

    # --- Model, Loss, Optimizer ---
    model = HydraNetSegmentationModel(num_classes=args.num_classes).to(device)
    criterion = SegmentationLoss(num_classes=args.num_classes, dice_weight=args.dice_weight).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # --- Training Execution ---
    best_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0

        for batch_idx, (images, masks) in enumerate(train_loader):
            images = images.to(device)
            masks = masks.to(device, dtype=torch.long)

            optimizer.zero_grad()
            preds = model(images)
            loss = criterion(preds, masks)
            
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            
            if batch_idx % args.print_freq == 0:
                print(f"[Epoch {epoch}/{args.epochs}] Batch {batch_idx} | Seg Loss: {loss.item():.4f}")

        avg_loss = epoch_loss / len(train_loader)
        print(f"--- Epoch {epoch} Completed | Average Loss: {avg_loss:.4f} ---")

        # Save Best Model
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_path = os.path.join(args.save_dir, "best_segmentation_model.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, save_path)
            print(f"✅ New best model saved to: {save_path}")

    print("\n🎉 Semantic Segmentation Training Successfully Finished!")

if __name__ == "__main__":
    main()