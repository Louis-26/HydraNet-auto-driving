import os
import sys
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
print(f"🔍 Current Script Directory: {CURRENT_DIR}")
print(f"🔍 Project Root Directory: {PROJECT_ROOT}")
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
        # CrossEntropyLoss automatically applies Log-Softmax
        # Expects raw logits of shape [B, C, H, W] and targets of shape [B, H, W]
        self.ce = nn.CrossEntropyLoss(ignore_index=255) 
        self.dice_weight = dice_weight
        self.eps = eps
        self.num_classes = num_classes

    def forward(self, preds, targets):
        # 1. Pixel-wise Cross Entropy Loss
        ce_loss = self.ce(preds, targets)

        # 2. Class-wise Dice Loss
        # Apply softmax to get probabilities for Dice computation
        preds_soft = torch.softmax(preds, dim=1) 
        
        # Convert targets to one-hot encoding: [B, H, W] -> [B, C, H, W]
        # Ignore index 255 needs to be masked out safely
        valid_mask = (targets != 255).unsqueeze(1).float()
        targets_safe = torch.where(targets == 255, torch.zeros_like(targets), targets)
        targets_one_hot = F.one_hot(targets_safe, num_classes=self.num_classes).permute(0, 3, 1, 2).float()

        # Calculate Intersection and Union over spatial dimensions (H, W)
        intersection = torch.sum(preds_soft * targets_one_hot * valid_mask, dim=(2, 3))
        union = torch.sum(preds_soft * valid_mask, dim=(2, 3)) + torch.sum(targets_one_hot * valid_mask, dim=(2, 3))
        
        # Compute Dice Loss and average across batches and classes
        dice_loss = 1.0 - (2.0 * intersection + self.eps) / (union + self.eps)
        dice_loss = dice_loss.mean()

        # 3. Total Loss
        return ce_loss + self.dice_weight * dice_loss

# ==========================================
# 🚀 Main Training Loop
# ==========================================
def main():
    print("="*50)
    print("🚀 Initiating Semantic Segmentation Training (Dummy Run)")
    print("="*50)

    # 1. Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 7  # Based on your 7-class KITTI mapping scheme
    epochs = 50
    learning_rate = 1e-4

    # 2. Dataloader Setup (Pointing to dummy_data for pipeline verification)
    train_img_dir = os.path.join(PROJECT_ROOT, "dummy_data", "ss", "train", "images")
    train_mask_dir = os.path.join(PROJECT_ROOT, "dummy_data", "ss", "train", "labels")
    
    train_dataset = KittiSegmentationDataset(
        image_dir=train_img_dir, 
        mask_dir=train_mask_dir, 
        target_size=(192, 640)
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=2, 
        shuffle=True, 
        drop_last=False
    )

    # 3. Initialize Model, Loss, and Optimizer
    model = HydraNetSegmentationModel(num_classes=num_classes).to(device)
    criterion = SegmentationLoss(num_classes=num_classes, dice_weight=1.0).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    # 4. Training Execution
    os.makedirs(os.path.join(PROJECT_ROOT, "checkpoints", "runs"), exist_ok=True)
    best_loss = float('inf')

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0

        for batch_idx, (images, masks) in enumerate(train_loader):
            images = images.to(device)
            # Ensure masks are LongTensors for CrossEntropy computation
            masks = masks.to(device, dtype=torch.long)

            optimizer.zero_grad()
            
            # Forward pass: Output shape should be [B, num_classes, H, W]
            preds = model(images)
            
            # Calculate Combined Loss (CE + Dice)
            loss = criterion(preds, masks)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            
            if batch_idx % 5 == 0:
                print(f"[Epoch {epoch}/{epochs}] Batch {batch_idx} | Seg Loss: {loss.item():.4f}")

        avg_loss = epoch_loss / len(train_loader)
        print(f"--- Epoch {epoch} Completed | Average Loss: {avg_loss:.4f} ---")

        # 5. Save the best model weights
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_path = os.path.join(PROJECT_ROOT, "checkpoints", "runs", "best_segmentation_model.pth")
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