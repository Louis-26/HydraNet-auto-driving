import os
import sys
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# ==========================================
# Bulletproof Path Resolution
# ==========================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dataloaders.kitti_dataset import KittiDepthDataset
from models.model import HydraNetDepthModel

def parse_args():
    parser = argparse.ArgumentParser(description="HydraNet Depth Estimation Training")
    parser.add_argument("--data_root", type=str, default=os.path.join(PROJECT_ROOT, "dummy_data", "de"), help="Root directory")
    parser.add_argument("--img_h", type=int, default=192, help="Input image height")
    parser.add_argument("--img_w", type=int, default=640, help="Input image width")
    parser.add_argument("--epochs", type=int, default=100, help="Total training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size per GPU")
    parser.add_argument("--lr", type=float, default=1e-4, help="Initial learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay for optimizer")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Computation device")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers")
    parser.add_argument("--pin_memory", action="store_true", help="Pin memory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save_dir", default=os.path.join(PROJECT_ROOT, "checkpoints", "runs", "official"), help="Save directory")
    parser.add_argument("--print_freq", type=int, default=5, help="Batch print frequency")
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

    print("="*55)
    print("🚀 Initiating Depth Estimation Training Pipeline (Masked Loss)")
    print("="*55)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    train_img_dir = os.path.join(args.data_root, "train", "images")
    train_depth_dir = os.path.join(args.data_root, "train", "depth")
    os.makedirs(args.save_dir, exist_ok=True)

    train_dataset = KittiDepthDataset(
        image_dir=train_img_dir, 
        depth_dir=train_depth_dir, 
        target_size=(args.img_h, args.img_w)
    )
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, 
        num_workers=args.num_workers, pin_memory=args.pin_memory, drop_last=False
    )

    model = HydraNetDepthModel().to(device)
    criterion = nn.L1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_loss = float('inf')
    best_model_path = os.path.join(args.save_dir, "best_depth_model.pth")

    model.train()
    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0.0
        
        for batch_idx, (images, depths) in enumerate(train_loader):
            images = images.to(device)
            depths = depths.to(device)
            
            optimizer.zero_grad()
            preds = model(images)
            
            # ==========================================
            # 🛡️ Only evaluate on non-zero entries
            # ==========================================
            valid_mask = depths > 0
            if valid_mask.sum() > 0:
                loss = criterion(preds[valid_mask], depths[valid_mask])
            else:
                loss = torch.tensor(0.0, device=device, requires_grad=True)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            if batch_idx % args.print_freq == 0:
                print(f"[Epoch {epoch}/{args.epochs}] Batch {batch_idx} | L1 Loss: {loss.item():.4f}", end='\r')
            
        avg_loss = epoch_loss / len(train_loader)
        print(f"\n--- Epoch {epoch} Completed | Average L1 Loss: {avg_loss:.4f} ---")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, best_model_path)
            print(f"✅ New best model saved to: {best_model_path}")

    print("\n🎉 Depth Estimation Training Successfully Finished!")

if __name__ == "__main__":
    main()