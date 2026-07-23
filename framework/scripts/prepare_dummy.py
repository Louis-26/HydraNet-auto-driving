import os
import sys
import shutil

# ==========================================
# 🚨 Bulletproof Path Resolution
# ==========================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

# Configure the base data root
DATA_ROOT = os.path.join(PROJECT_ROOT, "data")
DUMMY_DATA_ROOT = os.path.join(PROJECT_ROOT, "dummy_data")

# Map tasks to their actual raw dataset folder names
TASK_TO_RAW_FOLDER = {
    "od": "kitti_object",
    "ss": "kitti_semantics",
    "de": "kitti_depth"
}

def partition_dummy_data(task_name, folders_to_copy, train_n=7, val_n=2, test_n=1):
    """
    Extracts a sequential chunk of images from the raw 'train' split and partitions 
    them into dummy train, val, and test sets. Sourcing all from raw 'train' ensures 
    that Ground Truth labels safely exist for all dummy splits (especially val and test).
    """
    print(f"\n--- Partitioning Task: {task_name.upper()} ({train_n} Train, {val_n} Val, {test_n} Test) ---")
    
    raw_folder_name = TASK_TO_RAW_FOLDER[task_name]
    raw_train_dir = os.path.join(DATA_ROOT, raw_folder_name, "train")
    
    raw_images_dir = os.path.join(raw_train_dir, "images")
    if not os.path.exists(raw_images_dir):
        print(f"⚠️ Source directory not found: {raw_images_dir}. Skipping...")
        return
        
    all_images = sorted([f for f in os.listdir(raw_images_dir) if f.endswith(('.png', '.jpg'))])
    total_needed = train_n + val_n + test_n
    
    if len(all_images) < total_needed:
        print(f"⚠️ Not enough images in {raw_images_dir}. Found {len(all_images)}, need {total_needed}.")
        return
        
    # ✂️ Slice the images into the precise splits requested
    splits = {
        "train": all_images[:train_n],
        "val": all_images[train_n : train_n + val_n],
        "test": all_images[train_n + val_n : total_needed]
    }
    
    for split_name, img_list in splits.items():
        dummy_split_dir = os.path.join(DUMMY_DATA_ROOT, task_name, split_name)
        
        for folder_name in folders_to_copy:
            src_dir = os.path.join(raw_train_dir, folder_name)
            dst_dir = os.path.join(dummy_split_dir, folder_name)
            
            os.makedirs(dst_dir, exist_ok=True)
            
            success_count = 0
            for img_filename in img_list:
                base_name = os.path.splitext(img_filename)[0]
                
                # Smart extension resolution based on folder type
                if folder_name == 'YOLO_labels':
                    target_filename = base_name + '.txt'
                elif folder_name in ['semantic', 'semantic_rgb', 'depth']:
                    target_filename = base_name + '.png' 
                else:
                    target_filename = img_filename
                    
                src_file = os.path.join(src_dir, target_filename)
                dst_file = os.path.join(dst_dir, target_filename)
                
                # Fallback for semantic/depth in case they share the exact image extension
                if not os.path.exists(src_file) and folder_name != 'YOLO_labels':
                    src_file = os.path.join(src_dir, img_filename)
                    dst_file = os.path.join(dst_dir, img_filename)
                    
                if os.path.exists(src_file):
                    shutil.copy(src_file, dst_file)
                    success_count += 1
                    
            print(f"✅ Copied {success_count}/{len(img_list)} files into {task_name}/{split_name}/{folder_name}")

def main():
    print("="*50)
    print("🚀 Initiating Dummy Dataset Partitioning (7/2/1)")
    print("="*50)
    
    # 1. Object Detection (OD)
    partition_dummy_data(
        task_name="od", 
        folders_to_copy=["images", "annotated_images", "YOLO_labels"], 
        train_n=7, val_n=2, test_n=1
    )
    
    # 2. Semantic Segmentation (SS)
    partition_dummy_data(
        task_name="ss", 
        folders_to_copy=["images", "semantic", "semantic_rgb"], 
        train_n=7, val_n=2, test_n=1
    )
    
    # 3. Depth Estimation (DE)
    partition_dummy_data(
        task_name="de", 
        folders_to_copy=["images", "depth"], 
        train_n=7, val_n=2, test_n=1
    )
    
    print("\n" + "="*50)
    print("🎉 Dummy Dataset Preparation Complete!")
    print("="*50)

if __name__ == "__main__":
    main()