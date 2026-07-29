import os
import sys
import argparse
import random
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# ==========================================
# 🚨 CRITICAL FIX: Bulletproof Path Resolution
# ==========================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dataloaders.kitti_dataset import KittiDetectionDataset, kitti_collate_fn
from models.model import HydraNetDetectionModel
from utils.loss import YOLOLoss

# ==========================================
# Argument Parser
# ==========================================
def parse_args():
    parser = argparse.ArgumentParser(description="HydraNet Object Detection Training")

    # Dataset & Dimensions
    parser.add_argument("--data_root", type=str, default=os.path.join(PROJECT_ROOT, "dummy_data", "od"), help="Root directory of dataset")
    parser.add_argument("--num_classes", type=int, default=7, help="Number of detection classes")
    parser.add_argument("--img_h", type=int, default=192, help="Input image height")
    parser.add_argument("--img_w", type=int, default=640, help="Input image width")

    # Training Hyperparameters
    parser.add_argument("--epochs", type=int, default=100, help="Total training epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size per GPU")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay for optimizer")

    # Runtime & Checkpoint
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Computation device")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of dataloader workers")
    parser.add_argument("--pin_memory", action="store_true", help="Pin memory for faster data transfer")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--save_dir", default=os.path.join(PROJECT_ROOT, "checkpoints", "runs", "dummy"), help="Directory to save weights")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--print_freq", type=int, default=10, help="Frequency of printing epoch results")

    return parser.parse_args()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def main():
    args = parse_args()
    set_seed(args.seed)

    print("="*45)
    print("🚀 Initiating Training & Validation Pipeline...")
    print("="*45)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Dataset Paths ---
    train_img_dir = os.path.join(args.data_root, "train", "images")
    train_lbl_dir = os.path.join(args.data_root, "train", "YOLO_labels")
    val_img_dir = os.path.join(args.data_root, "val", "images")
    val_lbl_dir = os.path.join(args.data_root, "val", "YOLO_labels")
    os.makedirs(args.save_dir, exist_ok=True)

    # --- DataLoaders ---
    train_dataset = KittiDetectionDataset(image_dir=train_img_dir, label_dir=train_lbl_dir, target_size=(args.img_h, args.img_w))
    val_dataset = KittiDetectionDataset(image_dir=val_img_dir, label_dir=val_lbl_dir, target_size=(args.img_h, args.img_w))

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, 
        num_workers=args.num_workers, pin_memory=args.pin_memory, collate_fn=kitti_collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False, 
        num_workers=args.num_workers, pin_memory=args.pin_memory, collate_fn=kitti_collate_fn
    )

    # --- Model, Loss, Optimizer ---
    model = HydraNetDetectionModel(num_classes=args.num_classes).to(device)
    criterion = YOLOLoss(num_classes=args.num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # --- Resume ---
    best_val_loss = float("inf")
    start_epoch = 0

    if args.resume is not None:
        print(f"Loading checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        best_val_loss = checkpoint["best_val_loss"]
        start_epoch = checkpoint["epoch"] + 1
        print(f"Resume from epoch {start_epoch}")

    # --- Training Loop ---
    for epoch in range(start_epoch, args.epochs):
        
        # 1. Train
        model.train()
        train_loss = 0.0
        
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            
            optimizer.zero_grad()
            preds = model(images)
            loss = criterion(preds, targets)
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # 2. Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(device), targets.to(device)
                preds = model(images)
                loss = criterion(preds, targets)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        if (epoch + 1) % args.print_freq == 0:
            print(f"Epoch [{epoch+1}/{args.epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # 3. Save Best
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = os.path.join(args.save_dir, "best_detection_model.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
            }, save_path)
            print(f"🌟 Best model updated (Epoch {epoch+1}, Val Loss {best_val_loss:.4f})")

    print("="*45)
    print("✅ Training Complete.")
    print("="*45)

if __name__ == "__main__":
    main()