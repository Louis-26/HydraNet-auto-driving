import os
import sys
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

def main():
    print("=============================================")
    print("🚀 Initiating Training & Validation Pipeline...")
    print("=============================================")

    # 1. Hardware Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Data Pipeline Assembly (Train & Val)
    train_img_dir = os.path.join(PROJECT_ROOT, "dummy_data", "od", "train", "images")
    train_lbl_dir = os.path.join(PROJECT_ROOT, "dummy_data", "od", "train", "YOLO_labels")
    
    val_img_dir = os.path.join(PROJECT_ROOT, "dummy_data", "od", "val", "images")
    val_lbl_dir = os.path.join(PROJECT_ROOT, "dummy_data", "od", "val", "YOLO_labels")

    # Checkpoint save directory
    ckpt_dir = os.path.join(PROJECT_ROOT, "checkpoints", "runs")
    os.makedirs(ckpt_dir, exist_ok=True)

    train_dataset = KittiDetectionDataset(image_dir=train_img_dir, label_dir=train_lbl_dir, target_size=(192, 640))
    val_dataset = KittiDetectionDataset(image_dir=val_img_dir, label_dir=val_lbl_dir, target_size=(192, 640))
    
    train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=kitti_collate_fn)
    # Val dataloader typically doesn't need shuffling, batch_size can be 1 for precise metric tracking
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=kitti_collate_fn)

    # 3. Model, Loss, and Optimizer
    model = HydraNetDetectionModel(num_classes=7).to(device)
    criterion = YOLOLoss(num_classes=7).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # 4. The Training & Validation Loop
    num_epochs = 100
    best_val_loss = float('inf') # Initialize with infinity
    
    for epoch in range(num_epochs):
        # =========================
        #      TRAINING PHASE
        # =========================
        model.train() # Set model to training mode (enables Dropout, updates BatchNorm)
        train_epoch_loss = 0.0
        
        for images, targets in train_dataloader:
            images, targets = images.to(device), targets.to(device)
            # ================= 🚨 X光透视拦截点 =================
            # print("\n🔍 [DEBUG] targets shape:", targets.shape)
            # print("🔍 [DEBUG] targets 内容:\n", targets)
            # sys.exit(0)  # 打印完直接杀掉程序，看一眼就够了
            # ====================================================
            optimizer.zero_grad()
            preds = model(images)
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()
            train_epoch_loss += loss.item()
            
        avg_train_loss = train_epoch_loss / len(train_dataloader)

        # =========================
        #     VALIDATION PHASE
        # =========================
        model.eval() # Set model to evaluation mode (freezes BatchNorm/Dropout)
        val_epoch_loss = 0.0
        
        with torch.no_grad(): # CRITICAL: Disable gradient tracking to save memory and compute
            for val_images, val_targets in val_dataloader:
                val_images, val_targets = val_images.to(device), val_targets.to(device)
                val_preds = model(val_images)
                val_loss = criterion(val_preds, val_targets)
                val_epoch_loss += val_loss.item()
                
        avg_val_loss = val_epoch_loss / len(val_dataloader)

        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # =========================
        #     MODEL CHECKPOINTING
        # =========================
        # If the model performs better on Val, save it as the new best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = os.path.join(ckpt_dir, "best_detection_model.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
            }, save_path)
            
            # Print a small notification when a new record is set
            if (epoch + 1) % 10 == 0:
                 print(f"  --> 🌟 New Best Model Saved! (Val Loss: {best_val_loss:.4f})")

    print("=============================================")
    print("✅ Training complete! Best model weights secured.")
    print("=============================================")

if __name__ == "__main__":
    main()