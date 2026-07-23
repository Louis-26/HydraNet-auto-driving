import os
import sys
import cv2
import argparse

# ==========================================
# 🚨 Bulletproof Path Resolution
# ==========================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

def visualize_bboxes(img, bboxes, labels, class_names):
    """
    Draws bounding boxes and corresponding label names on the original image for visualization.
    
    Args:
        img (numpy.ndarray): The image array (BGR format from cv2).
        bboxes (list or ndarray): List of YOLO format bounding boxes [[cx, cy, w, h], ...].
        labels (list or ndarray): List of class IDs corresponding to the bboxes.
        class_names (dict): Dictionary mapping class IDs to string names.
        
    Returns:
        numpy.ndarray: A new image array with annotations drawn.
    """
    # Create a copy so we don't modify the original array in-place
    annotated_img = img.copy()
    img_h, img_w = annotated_img.shape[:2]
    
    for bbox, class_id in zip(bboxes, labels):
        cx_norm, cy_norm, w_norm, h_norm = bbox
        
        # Convert YOLO normalized coordinates back to absolute pixels
        cx = cx_norm * img_w
        cy = cy_norm * img_h
        bw = w_norm * img_w
        bh = h_norm * img_h
        
        # Calculate top-left (x1, y1) and bottom-right (x2, y2)
        x1 = int(cx - bw / 2)
        y1 = int(cy - bh / 2)
        x2 = int(cx + bw / 2)
        y2 = int(cy + bh / 2)
        
        # Draw bounding box (Green)
        color = (0, 255, 0) 
        thickness = 2
        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, thickness)
        
        # Fetch class name and setup font
        label_name = class_names.get(int(class_id), f"Class_{int(class_id)}")
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        
        # Get text size to draw a solid background rectangle for readability
        (text_w, text_h), baseline = cv2.getTextSize(label_name, font, font_scale, 1)
        cv2.rectangle(annotated_img, (x1, y1 - text_h - baseline), (x1 + text_w, y1), color, -1)
        
        # Overlay the text (Black text on Green background)
        cv2.putText(annotated_img, label_name, (x1, y1 - baseline), font, font_scale, (0, 0, 0), 1)
        
    return annotated_img

def main():
    # ==========================================
    # 🎛️ CLI Argument Parsing
    # ==========================================
    parser = argparse.ArgumentParser(description="Visualize YOLO bounding boxes on Object Detection datasets.")
    parser.add_argument(
        "--dataset_dir", 
        type=str, 
        default=os.path.join(PROJECT_ROOT, "dummy_data", "od"),
        help="Path to the root of the Object Detection dataset (must contain train/val/test splits)."
    )
    args = parser.parse_args()

    print("="*50)
    print("🚀 Initiating Object Detection Dataset Annotation")
    print(f"📂 Targeting Dataset Directory: {args.dataset_dir}")
    print("="*50)

    # Dictionary mapping KITTI class IDs to string names
    KITTI_CLASSES = {
        0: 'Car',
        1: 'Van',
        2: 'Truck',
        3: 'Pedestrian',
        4: 'Person_sitting',
        5: 'Cyclist',
        6: 'Tram',
        7: 'Misc',
        8: 'DontCare'
    }

    # Process all splits in the provided dataset directory
    splits = ["train", "val", "test"]
    
    for split in splits:
        # Dynamically inject the parsed dataset directory here
        base_dir = os.path.join(args.dataset_dir, split)
        img_dir = os.path.join(base_dir, "images")
        label_dir = os.path.join(base_dir, "YOLO_labels")
        
        annotated_dir = os.path.join(base_dir, "annotated_images")
        
        # Check if the split exists before processing
        if not os.path.exists(img_dir) or not os.path.exists(label_dir):
            print(f"⚠️ Skipping '{split}' split, required directories not found at {base_dir}")
            continue
            
        os.makedirs(annotated_dir, exist_ok=True)
        images = [f for f in os.listdir(img_dir) if f.endswith(('.png', '.jpg'))]
        print(f"\nProcessing {len(images)} images in OD '{split}' split...")
        
        success_count = 0
        for img_name in images:
            base_name = os.path.splitext(img_name)[0]
            
            img_path = os.path.join(img_dir, img_name)
            label_path = os.path.join(label_dir, f"{base_name}.txt")
            
            # Load the original image
            img = cv2.imread(img_path)
            if img is None:
                print(f"  [Error] Could not read image: {img_path}")
                continue
                
            bboxes = []
            labels = []
            
            # Parse YOLO labels if the txt file exists
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            # YOLO format: class_id, x_center, y_center, width, height
                            labels.append(int(parts[0]))
                            bboxes.append([float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])])
            
            # Call our visualization function
            annotated_img = visualize_bboxes(img, bboxes, labels, KITTI_CLASSES)
            
            # Save to annotated_images folder
            output_path = os.path.join(annotated_dir, img_name)
            cv2.imwrite(output_path, annotated_img)
            success_count += 1
            
        print(f"✅ Generated {success_count} annotated images in: {annotated_dir}")

    print("\n" + "="*50)
    print("🎉 Dataset Annotation Complete! Check your 'annotated_images' folders.")
    print("="*50)

if __name__ == "__main__":
    main()