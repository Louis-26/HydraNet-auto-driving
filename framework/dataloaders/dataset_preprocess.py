import os
import cv2

# ==========================================
# CONFIGURATION
# ==========================================
# Full KITTI Classes for fine-grained detection
KITTI_CLASSES = {
    "Car": 0,
    "Van": 1,
    "Truck": 2,
    "Pedestrian": 3,
    "Person_sitting": 4,
    "Cyclist": 5,
    "Tram": 6
}

def convert_kitti_to_yolo(img_dir, src_label_dir, dst_label_dir):
    """
    Reads 15-dim KITTI labels, normalizes them using image dimensions,
    and writes 5-dim YOLO labels to a new directory.
    """
    os.makedirs(dst_label_dir, exist_ok=True)
    
    img_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.png')])
    processed_count = 0
    
    for img_name in img_files:
        base_name = os.path.splitext(img_name)[0]
        img_path = os.path.join(img_dir, img_name)
        src_lbl_path = os.path.join(src_label_dir, base_name + '.txt')
        dst_lbl_path = os.path.join(dst_label_dir, base_name + '.txt')
        
        # Read image strictly to get accurate original dimensions
        image = cv2.imread(img_path)
        if image is None:
            print(f"WARNING: Image {img_name} not found. Skipping.")
            continue
            
        orig_h, orig_w = image.shape[:2]
        yolo_lines = []
        
        if os.path.exists(src_lbl_path):
            with open(src_lbl_path, 'r') as f_in:
                for line in f_in.readlines():
                    parts = line.strip().split()
                    if len(parts) >= 8:
                        obj_type = parts[0]
                        if obj_type not in KITTI_CLASSES:
                            continue
                            
                        class_id = KITTI_CLASSES[obj_type]
                        xmin, ymin = float(parts[4]), float(parts[5])
                        xmax, ymax = float(parts[6]), float(parts[7])
                        
                        # YOLO Normalization (cx, cy, w, h)
                        cx = ((xmin + xmax) / 2.0) / orig_w
                        cy = ((ymin + ymax) / 2.0) / orig_h
                        w = (xmax - xmin) / orig_w
                        h = (ymax - ymin) / orig_h
                        
                        yolo_lines.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
        
        # Write to the new YOLO_labels directory
        with open(dst_lbl_path, 'w') as f_out:
            f_out.writelines(yolo_lines)
            
        processed_count += 1

    print(f"Successfully generated {processed_count} YOLO format labels in: {dst_label_dir}")

if __name__ == "__main__":
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
    
    print("Starting offline conversion from KITTI to YOLO format for the FULL dataset...")
    
    # Automatically process 'train', 'val', and 'test' splits
    splits = ["train", "val", "test"]
    
    for split in splits:
        print(f"\n--- Processing {split.upper()} split ---")
        
        # Target the actual full dataset directories
        IMG_DIR = os.path.join(PROJECT_ROOT, "data", "kitti_object", split, "images")
        SRC_LBL_DIR = os.path.join(PROJECT_ROOT, "data", "kitti_object", split, "labels")
        DST_LBL_DIR = os.path.join(PROJECT_ROOT, "data", "kitti_object", split, "YOLO_labels")
        
        if not os.path.exists(SRC_LBL_DIR):
            print(f"WARNING: Source label directory not found at {SRC_LBL_DIR}. Skipping.")
            continue
            
        # ==========================================
        # Idempotency Check: Skip if already converted
        # ==========================================
        if os.path.exists(DST_LBL_DIR):
            src_count = len([f for f in os.listdir(SRC_LBL_DIR) if f.endswith('.txt')])
            dst_count = len([f for f in os.listdir(DST_LBL_DIR) if f.endswith('.txt')])
            
            if src_count > 0 and src_count == dst_count:
                print(f"SKIP: '{split}' split is already converted ({dst_count} files match perfectly).")
                continue
                
        convert_kitti_to_yolo(IMG_DIR, SRC_LBL_DIR, DST_LBL_DIR)
        
    print("\nAll offline conversions completed successfully!")