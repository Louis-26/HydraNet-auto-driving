import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
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

def main():
    print("="*50)
    print("🚀 Initiating Depth Estimation Training (Dummy Run)")
    print("="*50)

    # 1. Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 2
    epochs = 50
    learning_rate = 1e-4

    # 2. Setup Dataloader
    # Note: Ensure the folder name for depth ground truth is 'labels' or adjust accordingly
    train_img_dir = os.path.join(PROJECT_ROOT, "dummy_data", "de", "train", "images")
    train_depth_dir = os.path.join(PROJECT_ROOT, "dummy_data", "de", "train", "labels")
    
    train_dataset = KittiDepthDataset(
        image_dir=train_img_dir, 
        depth_dir=train_depth_dir, 
        target_size=(192, 640)
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        drop_last=False
    )

    # 3. Initialize Model, Loss, and Optimizer
    model = HydraNetDepthModel().to(device)
    
    # L1 Loss (Mean Absolute Error) is highly robust for depth estimation tasks
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 4. Save Directory Setup
    save_dir = os.path.join(PROJECT_ROOT, "checkpoints", "runs")
    os.makedirs(save_dir, exist_ok=True)
    best_loss = float('inf')
    best_model_path = os.path.join(save_dir, "best_depth_model.pth")

    # 5. Training Loop
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        
        for batch_idx, (images, depths) in enumerate(train_loader):
            images = images.to(device)
            depths = depths.to(device)
            
            # Forward pass
            preds = model(images)
            
            # Calculate Loss
            loss = criterion(preds, depths)
            
            # Backward pass & optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            print(f"[Epoch {epoch+1}/{epochs}] Batch {batch_idx} | L1 Loss: {loss.item():.4f}", end='\r')
            
        avg_loss = epoch_loss / len(train_loader)
        print(f"--- Epoch {epoch+1} Completed | Average L1 Loss: {avg_loss:.4f} ---")
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, best_model_path)
            print(f"✅ New best model saved to: {best_model_path}")

    print("\n🎉 Depth Estimation Training Successfully Finished!")

if __name__ == "__main__":
    main()