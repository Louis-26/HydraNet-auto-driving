import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

class KittiDetectionDataset(Dataset):
    """
    Custom Dataset tailored for the HydraNet YOLOv8 detection branch.
    Specifically designed to read images and labels from the dummy_data directory.
    """
    def __init__(self, image_dir, label_dir=None, target_size=(192, 640)):
        """
        :param image_dir: Directory path containing the images.
        :param label_dir: Directory path containing the .txt labels (YOLO format: class cx cy w h).
        :param target_size: (H, W) corresponding to the 192x640 input requirement from the project report.
        """
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.target_size = target_size # (H, W) -> (192, 640)
        
        # Retrieve all image filenames (supporting both .png and .jpg)
        self.image_files = sorted([
            f for f in os.listdir(image_dir) 
            if f.endswith('.png') or f.endswith('.jpg')
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        # 1. Read and process the image
        img_name = self.image_files[index]
        img_path = os.path.join(self.image_dir, img_name)
        
        # OpenCV reads in BGR by default; convert to RGB
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Store original dimensions in case bounding box restoration is needed later
        orig_h, orig_w = image.shape[:2]
        
        # Resize to (640, 192) - Note that cv2.resize expects (Width, Height)
        image = cv2.resize(image, (self.target_size[1], self.target_size[0]))
        
        # Normalize pixel values to [0, 1]
        image = image.astype(np.float32) / 255.0
        # Apply standard preprocessing (converting to PyTorch Tensor format: C, H, W)
        image = torch.from_numpy(image).permute(2, 0, 1)

        # 2. Read and process labels (if they exist)
        labels = []
        if self.label_dir is not None:
            # Assume label and image filenames match exactly, differing only by extension
            label_name = os.path.splitext(img_name)[0] + '.txt'
            label_path = os.path.join(self.label_dir, label_name)
            
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    for line in f.readlines():
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            # Assume dummy_data strictly follows YOLO format: class_id, cx, cy, w, h
                            class_id = int(float(parts[0]))
                            cx = float(parts[1])
                            cy = float(parts[2])
                            w = float(parts[3])
                            h = float(parts[4])
                            labels.append([class_id, cx, cy, w, h])
        
        # Convert to Tensor; if no bounding boxes exist, return an empty tensor of shape (0, 5)
        labels = torch.tensor(labels, dtype=torch.float32)
        if labels.shape[0] == 0:
            labels = torch.zeros((0, 5), dtype=torch.float32)
            
        return image, labels


def kitti_collate_fn(batch):
    """
    Custom collate_fn, absolutely crucial!
    PyTorch cannot natively batch bounding box tensors of different sizes.
    This function injects the image index (batch_idx) into the first column of the label tensor.
    Final label shape: [N, 6] -> [batch_idx, class_id, cx, cy, w, h]
    """
    images, labels = zip(*batch)
    
    # Stack images together -> Shape: (Batch_Size, 3, 192, 640)
    images = torch.stack(images, 0)
    
    # Concatenate all labels and insert the batch_idx at the very beginning
    batch_labels = []
    for i, label in enumerate(labels):
        if label.shape[0] > 0:
            # Create a column filled with the current batch index (i)
            batch_idx = torch.full((label.shape[0], 1), i, dtype=torch.float32)
            # Concatenate to the front of the original label -> [batch_idx, class_id, cx, cy, w, h]
            label_with_batch = torch.cat((batch_idx, label), dim=1)
            batch_labels.append(label_with_batch)
            
    if len(batch_labels) > 0:
        targets = torch.cat(batch_labels, dim=0)
    else:
        # Fallback if there are absolutely no objects in the entire batch
        targets = torch.zeros((0, 6), dtype=torch.float32)
        
    return images, targets

if __name__ == "__main__":
    import shutil
    from torch.utils.data import DataLoader

    print("=============================================")
    print("🚀 Starting test for KittiDetectionDataset module...")
    print("=============================================")

    # 1. Automatically create a temporary sandbox directory to store dummy test data
    test_dir = "./temp_test_data"
    img_dir = os.path.join(test_dir, "images")
    lbl_dir = os.path.join(test_dir, "labels")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    try:
        # 2. Forge two dummy images with original KITTI dimensions (e.g., 375x1242)
        dummy_image = np.zeros((375, 1242, 3), dtype=np.uint8)
        cv2.imwrite(os.path.join(img_dir, "000000.png"), dummy_image)
        cv2.imwrite(os.path.join(img_dir, "000001.png"), dummy_image)

        # 3. Forge the corresponding pure YOLO format label files
        # The first image contains 2 cars
        with open(os.path.join(lbl_dir, "000000.txt"), "w") as f:
            f.write("0 0.50 0.50 0.20 0.30\n")
            f.write("0 0.70 0.80 0.15 0.15\n")
        
        # The second image contains 1 car
        with open(os.path.join(lbl_dir, "000001.txt"), "w") as f:
            f.write("0 0.30 0.30 0.10 0.20\n")

        # 4. Instantiate our Dataset
        dataset = KittiDetectionDataset(image_dir=img_dir, label_dir=lbl_dir, target_size=(192, 640))
        print(f"✅ Dataset loaded successfully. Found {len(dataset)} samples.")

        # 5. Instantiate the DataLoader, noting the use of our custom kitti_collate_fn
        dataloader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=kitti_collate_fn)

        # 6. Extract one Batch for ultimate validation
        for images, targets in dataloader:
            print("\n[Batch Data Shape Validation]")
            print(f"👉 Images Tensor Shape: {images.shape}")  
            print(f"👉 Targets Tensor Shape: {targets.shape}")
            
            print("\n[Targets Content Validation (Should contain batch_idx, class_id, cx, cy, w, h)]")
            print(targets)
            
            # Assertions - Using code to absolutely guarantee logic correctness
            assert images.shape == (2, 3, 192, 640), "❌ Image Resize or concatenation failed!"
            assert targets.shape == (3, 6), "❌ Bounding Box concatenation or Batch_idx insertion failed!"
            break  # Only test one batch

        print("\n🎉 Test passed perfectly! Dataset and Collate_fn logic are 100% correct.")

    finally:
        # 7. Environmental cleanup: Delete temporary files after testing to leave no garbage behind
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)