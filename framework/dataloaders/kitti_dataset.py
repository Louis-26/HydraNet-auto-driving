import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class KittiDetectionDataset(Dataset):
    """
    Custom Dataset tailored for the HydraNet YOLOv8 detection branch.
    Optimized to directly read pre-processed 5-dim YOLO format labels.
    """
    def __init__(self, image_dir, label_dir=None, target_size=(192, 640)):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.target_size = target_size # (H, W) -> (192, 640)
        
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
        
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Failed to read image at: {img_path}")
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize and normalize directly (No need to store original dimensions anymore)
        image = cv2.resize(image, (self.target_size[1], self.target_size[0]))
        image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)

        # 2. Read directly from pre-processed 5-dim YOLO_labels
        labels = []
        if self.label_dir is not None:
            label_name = os.path.splitext(img_name)[0] + '.txt'
            label_path = os.path.join(self.label_dir, label_name)
            
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    for line in f.readlines():
                        parts = line.strip().split()
                        
                        # Direct mapping from 5-dim YOLO txt
                        if len(parts) == 5:
                            class_id = float(parts[0])
                            cx = float(parts[1])
                            cy = float(parts[2])
                            w = float(parts[3])
                            h = float(parts[4])
                            
                            labels.append([class_id, cx, cy, w, h])
        
        labels = torch.tensor(labels, dtype=torch.float32)
        if labels.shape[0] == 0:
            labels = torch.zeros((0, 5), dtype=torch.float32)
            
        return image, labels
    
    
class KittiSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, target_size=(192, 640)):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.target_size = target_size
        
        self.image_files = sorted(os.listdir(image_dir))
        self.mask_files = sorted(os.listdir(mask_dir))
        
        # ==========================================
        # 🗺️ Raw KITTI (34 classes) to Custom (7 classes) Mapping
        # Target: 0:road, 1:car, 2:vegetation, 3:building, 4:sky, 5:sidewalk, 6:fence
        # ==========================================
        # Create a lookup table filled with ignore_index (255)
        self.mapping = np.full(256, 255, dtype=np.uint8)
        
        # Map specific raw KITTI IDs to our 7 target IDs
        self.mapping[7] = 0   # road -> road
        self.mapping[26] = 1  # car -> car
        self.mapping[21] = 2  # vegetation -> vegetation
        self.mapping[11] = 3  # building -> building
        self.mapping[23] = 4  # sky -> sky
        self.mapping[8] = 5   # sidewalk -> sidewalk
        self.mapping[13] = 6  # fence -> fence

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        mask_name = self.mask_files[idx]
        
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, mask_name)
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Read mask as grayscale (raw IDs)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Resize image (Bilinear)
        image = cv2.resize(image, (self.target_size[1], self.target_size[0]), interpolation=cv2.INTER_LINEAR)
        
        # Resize mask (Nearest Neighbor to prevent fractional IDs)
        mask = cv2.resize(mask, (self.target_size[1], self.target_size[0]), interpolation=cv2.INTER_NEAREST)
        
        # 🚀 Apply the mapping using NumPy advanced indexing (Extremely fast!)
        mask = self.mapping[mask]
        
        # Convert to Tensor
        image = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0
        mask = torch.from_numpy(mask).long()
        
        return image, mask
    

import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

class KittiDepthDataset(Dataset):
    """
    DataLoader for KITTI Depth Estimation.
    """
    def __init__(self, image_dir, depth_dir, target_size=(192, 640)):
        self.image_dir = image_dir
        self.depth_dir = depth_dir
        self.target_size = target_size
        
        self.image_files = sorted(os.listdir(image_dir))
        self.depth_files = sorted(os.listdir(depth_dir))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        depth_path = os.path.join(self.depth_dir, self.depth_files[idx])
        
        # Read and resize RGB image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.target_size[1], self.target_size[0]), interpolation=cv2.INTER_LINEAR)
        
        # Read Depth map (Using IMREAD_ANYDEPTH to support 16-bit PNGs if they exist)
        depth = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
        if depth is None:
            # Fallback for standard 8-bit dummy data
            depth = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
            
        # Nearest neighbor interpolation prevents creating fake continuous depth values at edges
        depth = cv2.resize(depth, (self.target_size[1], self.target_size[0]), interpolation=cv2.INTER_NEAREST)
        
        # Convert to float. (For raw KITTI, depth is often divided by 256.0, adjust if needed)
        depth = depth.astype(np.float32) / 255.0
        
        # Convert to Tensors: Image [3, H, W], Depth [1, H, W]
        image = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0
        depth = torch.from_numpy(depth).float().unsqueeze(0)
        
        return image, depth
    
def kitti_collate_fn(batch):
    images, labels = zip(*batch)
    images = torch.stack(images, 0)
    
    batch_labels = []
    for i, label in enumerate(labels):
        if label.shape[0] > 0:
            batch_idx = torch.full((label.shape[0], 1), i, dtype=torch.float32)
            label_with_batch = torch.cat((batch_idx, label), dim=1)
            batch_labels.append(label_with_batch)
            
    if len(batch_labels) > 0:
        targets = torch.cat(batch_labels, dim=0)
    else:
        targets = torch.zeros((0, 6), dtype=torch.float32)
        
    return images, targets


class KittiMultitaskDataset(Dataset):
    def __init__(self, data_root, split='train', target_size=(192, 640)):
        """
        Industry-standard Disjoint Multi-Task Dataset for HydraNet.
        Handles unaligned/separate datasets across tasks by dynamically routing 
        valid labels and returning 'ignore' tensors for missing tasks.
        """
        self.target_size = target_size
        self.samples = []
        
        # 34-to-7 Mapping for KITTI Semantics
        self.mapping = np.full(256, 255, dtype=np.uint8)
        self.mapping[[7, 26, 21, 11, 23, 8, 13]] = [0, 1, 2, 3, 4, 5, 6]
        
        # ==========================================
        # Indexing all completely separate sub-datasets
        # ==========================================
        
        # 1. Gather Object Detection (OD) Samples
        od_img_dir = os.path.join(data_root, "kitti_object", split, "images")
        od_lbl_dir = os.path.join(data_root, "kitti_object", split, "YOLO_labels")
        if os.path.exists(od_img_dir):
            for f in os.listdir(od_img_dir):
                if f.endswith(('.png', '.jpg')):
                    self.samples.append({'source_task': 'od', 'img_name': f, 'img_dir': od_img_dir, 'lbl_dir': od_lbl_dir})
                    
        # 2. Gather Semantic Segmentation (SS) Samples
        ss_img_dir = os.path.join(data_root, "kitti_semantics", split, "images")
        ss_lbl_dir = os.path.join(data_root, "kitti_semantics", split, "semantic")
        if os.path.exists(ss_img_dir):
            for f in os.listdir(ss_img_dir):
                if f.endswith(('.png', '.jpg')):
                    self.samples.append({'source_task': 'ss', 'img_name': f, 'img_dir': ss_img_dir, 'lbl_dir': ss_lbl_dir})
                    
        # 3. Gather Depth Estimation (DE) Samples
        de_img_dir = os.path.join(data_root, "kitti_depth", split, "images")
        de_lbl_dir = os.path.join(data_root, "kitti_depth", split, "depth")
        if os.path.exists(de_img_dir):
            for f in os.listdir(de_img_dir):
                if f.endswith(('.png', '.jpg')):
                    self.samples.append({'source_task': 'de', 'img_name': f, 'img_dir': de_img_dir, 'lbl_dir': de_lbl_dir})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_info = self.samples[idx]
        task = sample_info['source_task']
        img_name = sample_info['img_name']
        base_name = os.path.splitext(img_name)[0]
        
        # 1. Load Universal Image
        img_path = os.path.join(sample_info['img_dir'], img_name)
        image = cv2.imread(img_path)
        if image is None: raise ValueError(f"Failed to load: {img_path}")
        image = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), (self.target_size[1], self.target_size[0]))
        image_tensor = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0
        
        # 2. Initialize DEFAULT IGNORE variables for all 3 tasks
        od_targets = []
        ss_mask = np.full((self.target_size[0], self.target_size[1]), 255, dtype=np.uint8) # 255 = ignore
        de_mask = np.zeros((self.target_size[0], self.target_size[1]), dtype=np.float32)   # 0 = ignore
        
        # 3. Route specific Ground Truth based on the image's source task
        if task == 'od':
            txt_path = os.path.join(sample_info['lbl_dir'], f"{base_name}.txt")
            if os.path.exists(txt_path):
                with open(txt_path, 'r') as f:
                    for line in f.readlines():
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            od_targets.append([float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])])
                            
        elif task == 'ss':
            png_path = os.path.join(sample_info['lbl_dir'], f"{base_name}.png")
            if not os.path.exists(png_path): png_path = os.path.join(sample_info['lbl_dir'], img_name)
            if os.path.exists(png_path):
                raw_ss = cv2.imread(png_path, cv2.IMREAD_GRAYSCALE)
                if raw_ss is not None:
                    raw_ss = cv2.resize(raw_ss, (self.target_size[1], self.target_size[0]), interpolation=cv2.INTER_NEAREST)
                    ss_mask = self.mapping[raw_ss] # Apply 34-to-7 mapping
                    
        elif task == 'de':
            png_path = os.path.join(sample_info['lbl_dir'], f"{base_name}.png")
            if not os.path.exists(png_path): png_path = os.path.join(sample_info['lbl_dir'], img_name)
            if os.path.exists(png_path):
                raw_de = cv2.imread(png_path, cv2.IMREAD_ANYDEPTH)
                if raw_de is None: raw_de = cv2.imread(png_path, cv2.IMREAD_GRAYSCALE)
                if raw_de is not None:
                    raw_de = cv2.resize(raw_de, (self.target_size[1], self.target_size[0]), interpolation=cv2.INTER_NEAREST)
                    de_mask = raw_de.astype(np.float32) / 256.0

        # Convert to final tensors
        od_tensor = torch.tensor(od_targets, dtype=torch.float32)
        ss_tensor = torch.from_numpy(ss_mask).long()
        de_tensor = torch.from_numpy(de_mask).float().unsqueeze(0)
        
        return image_tensor, od_tensor, ss_tensor, de_tensor

def multitask_collate_fn(batch):
    images, od_targets, ss_masks, de_masks = zip(*batch)
    images = torch.stack(images, 0)
    ss_masks = torch.stack(ss_masks, 0)
    de_masks = torch.stack(de_masks, 0)
    return images, list(od_targets), ss_masks, de_masks

if __name__ == "__main__":
    print("=============================================")
    print("Starting real dummy_data test for KittiDetectionDataset...")
    print("=============================================")

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
    
    dummy_img_dir = os.path.join(PROJECT_ROOT, "dummy_data", "od", "train", "images")
    
    # POINT TO THE NEW DIRECTORY: YOLO_labels instead of labels
    dummy_lbl_dir = os.path.join(PROJECT_ROOT, "dummy_data", "od", "train", "YOLO_labels")

    if not os.path.exists(dummy_img_dir):
        raise FileNotFoundError(f"Dummy data not found at: {dummy_img_dir}. Did you run prepare_dummy.py?")

    # 1. Instantiate the Dataset
    dataset = KittiDetectionDataset(image_dir=dummy_img_dir, label_dir=dummy_lbl_dir, target_size=(192, 640))
    print(f"Dataset loaded successfully. Found {len(dataset)} real dummy samples.")

    # 2. Instantiate the DataLoader
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=kitti_collate_fn)

    # 3. Grab a batch to verify the parsed results
    for images, targets in dataloader:
        print("\n[Batch Data Shape Validation]")
        print(f"Images Tensor Shape: {images.shape}")  
        print(f"Targets Tensor Shape: {targets.shape}")
        
        print("\n[Targets Content Validation (Directly from YOLO_labels!)]")
        print("Columns: [batch_idx, class_id, cx, cy, w, h]")
        print(targets)
        
        assert images.shape == (2, 3, 192, 640), "Image Resize or concatenation failed!"
        assert targets.shape[1] == 6, "Target columns format failed!"
        
        break

    print("\nTest passed perfectly! Direct YOLO label reading is 100% functional.")