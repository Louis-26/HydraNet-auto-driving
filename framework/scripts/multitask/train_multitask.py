import os
import sys
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

# ==========================================
# Bulletproof Path Resolution
# ==========================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dataloaders.kitti_dataset import KittiMultitaskDataset, multitask_collate_fn
from models.model import HydraNetMultitaskModel
from utils.loss import YOLOLoss

HAS_YOLO_LOSS = True

# ==========================================
# Argument Parser
# ==========================================
def parse_args():
    parser = argparse.ArgumentParser(description="HydraNet 3-Task Joint Training Pipeline")
    
    # Dataset & Paths
    parser.add_argument('--data_root', type=str, default=os.path.join(PROJECT_ROOT, "data"), help='Root directory containing official multitask dataset')
    parser.add_argument('--save_dir', type=str, default=os.path.join(PROJECT_ROOT, "checkpoints", "runs", "official"), help='Directory to save the best model weights')
    
    # Training Hyperparameters
    parser.add_argument('--epochs', type=int, default=50, help='Total number of training epochs')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size per GPU')
    parser.add_argument('--lr', type=float, default=1e-4, help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay for optimizer')
    
    # Multi-Task Loss Weighting (Lambdas)
    parser.add_argument('--lambda_od', type=float, default=1.0, help='Weight for Object Detection Loss')
    parser.add_argument('--lambda_ss', type=float, default=1.0, help='Weight for Semantic Segmentation Loss')
    parser.add_argument('--lambda_de', type=float, default=2.0, help='Weight for Depth Estimation Loss')
    
    # Runtime & Reproducibility
    parser.add_argument('--device', default="cuda", choices=["cuda", "cpu"], help="Computation device")
    parser.add_argument('--num_workers', type=int, default=4, help="Number of dataloader workers")
    parser.add_argument('--pin_memory', action="store_true", default=True, help="Pin memory for faster data transfer")
    parser.add_argument('--print_freq', type=int, default=10, help="Batch printing frequency")
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


# ==========================================
# Main Training Loop
# ==========================================
def main():
    args = parse_args()
    set_seed(args.seed)
    
    print("="*60)
    print("🚀 Initiating HydraNet 3-Task Joint Training Pipeline")
    print(f"📂 Dataset Root  : {args.data_root}")
    print(f"⚖️  Loss Weights  : OD({args.lambda_od}) | SS({args.lambda_ss}) | DE({args.lambda_de})")
    print("="*60)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Dataloader Setup
    train_dataset = KittiMultitaskDataset(args.data_root, split='train')
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        collate_fn=multitask_collate_fn,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory
    )

    # Model Initialization
    model = HydraNetMultitaskModel(num_ss_classes=7, num_od_classes=7).to(device)
    
    # Criteria and Optimizer
    criterion_ss = nn.CrossEntropyLoss(ignore_index=255)
    criterion_de = nn.L1Loss() 
    
    if HAS_YOLO_LOSS:
        criterion_od = YOLOLoss(num_classes=7).to(device)
    else:
        print("⚠️ [Warning] 'YOLOLoss' not found in utils.loss.")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    os.makedirs(args.save_dir, exist_ok=True)
    best_loss = float('inf')

    model.train()
    for epoch in range(args.epochs):
        epoch_total = 0.0
        epoch_od, epoch_ss, epoch_de = 0.0, 0.0, 0.0
        
        for batch_idx, (images, od_targets, ss_masks, de_masks) in enumerate(train_loader):
            images = images.to(device)
            ss_masks = ss_masks.to(device)
            de_masks = de_masks.to(device)
            
            optimizer.zero_grad()
            out_od, out_ss, out_de = model(images)
            
            # --- SS & DE Loss Computation ---
            valid_ss_pixels = (ss_masks != 255).sum()
            loss_ss = criterion_ss(out_ss, ss_masks) if valid_ss_pixels > 0 else torch.tensor(0.0, device=device, requires_grad=True)
            
            valid_depth_mask = de_masks > 0
            loss_de = criterion_de(out_de[valid_depth_mask], de_masks[valid_depth_mask]) if valid_depth_mask.sum() > 0 else torch.tensor(0.0, device=device, requires_grad=True)
                
            # --- Robust Target Reshaping for Object Detection ---
            if HAS_YOLO_LOSS and out_od is not None:
                batched_targets = []
                if isinstance(od_targets, (list, tuple)):
                    for b_idx, target in enumerate(od_targets):
                        if len(target) > 0:
                            target = target.to(device)
                            # If target format is 5 columns [class_id, cx, cy, w, h], prepend the batch index column
                            if target.shape[1] == 5:
                                b_col = torch.full((target.shape[0], 1), b_idx, dtype=torch.float32, device=device)
                                batched_target = torch.cat((b_col, target), dim=1)
                                batched_targets.append(batched_target)
                            elif target.shape[1] == 6:
                                batched_targets.append(target)
                
                if len(batched_targets) > 0:
                    target_tensor = torch.cat(batched_targets, dim=0)
                    loss_od = criterion_od(out_od, target_tensor)
                else:   
                    loss_od = torch.tensor(0.0, device=device, requires_grad=True)
            else:
                loss_od = torch.tensor(0.0, device=device, requires_grad=True)
                
            # --- Joint Multi-Task Loss Equation ---
            total_loss = (args.lambda_od * loss_od) + \
                         (args.lambda_ss * loss_ss) + \
                         (args.lambda_de * loss_de)
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            
            epoch_total += total_loss.item()
            epoch_od += loss_od.item()
            epoch_ss += loss_ss.item()
            epoch_de += loss_de.item()
            
            if batch_idx % args.print_freq == 0:
                print(f"[Epoch {epoch+1}/{args.epochs}] Batch {batch_idx+1}/{len(train_loader)} | "
                      f"Total Loss: {total_loss.item():.4f} "
                      f"(OD: {loss_od.item():.4f} | SS: {loss_ss.item():.4f} | DE: {loss_de.item():.4f})", end='\r')
            
        # Epoch Aggregation
        avg_total = epoch_total / len(train_loader)
        avg_od = epoch_od / len(train_loader)
        avg_ss = epoch_ss / len(train_loader)
        avg_de = epoch_de / len(train_loader)
        
        # Step the learning rate scheduler
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(avg_total)
        new_lr = optimizer.param_groups[0]['lr']
        
        print(f"\n--- Epoch {epoch+1} Summary | Avg Total Loss: {avg_total:.4f} "
              f"[OD: {avg_od:.4f}, SS: {avg_ss:.4f}, DE: {avg_de:.4f}] ---")
              
        if new_lr < current_lr:
            print(f"📉 Learning Rate reduced from {current_lr} to {new_lr}")
        
        # Save Best Model Checkpoint
        if avg_total < best_loss:
            best_loss = avg_total
            save_path = os.path.join(args.save_dir, "best_multitask_model.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss
            }, save_path)
            print(f"🏆 New best multi-task model saved with loss: {best_loss:.4f}")

    print("\n" + "="*60)
    print(f"🎉 Multi-Task Training Complete! Best weights at: {args.save_dir}")
    print("="*60)

if __name__ == "__main__":
    main()