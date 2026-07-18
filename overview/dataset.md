# KITTI dataset
## Dataset structure
dataset/ 
├── kitti_object/
|   ├── training/
|   │   ├── images/      
│   |   └── labels/       
│	├── testing/
|   │   └── images/      
├── kitti_semantics
|   ├── training/
|   │   ├── images/      
│   |   ├── instance/       
│   |   ├── semantic/       
│   |   └── semantic_rgb/       
│	├── testing/
|   │   └── images/   
├── kitti_depth/
|   ├── train/
|   ├── train/
|   └── val/ 
├── kitti_raw/
|   ├── train/
|   ├── train/
|   └── val/ 

## Dataset overview

## KITTI Depth Estimation
depth&raw data:     
"2011_09_26_drive_0001"   # City
"2011_09_26_drive_0019"   # Residential
"2011_09_26_drive_0015"   # Road
"2011_09_28_drive_0016"   # Campus
"2011_09_28_drive_0053"   # Person


## KITTI Segmentation
all available on KITTI

## KITTI Object Detection
include all folders available on KITTI, including images and labels for each folder

original label(15 columns): 
- basic properties
    - column 1: class type(string, e.g. Car, Pedestrian, Cyclist)
    - column 2: truncated rate(float from 0-1, 0 means whole object is in image, 1 means object is fully truncated)
    - column 3: occluded rate(integer from 0-3, 0 means whole object is visible, 3 means object is fully occluded)
    - column 4: alpha(float from -pi to pi, observation angle of object)
- 2D bounding box
    - column 5: bbox_left (xmin), upper left pixel coordinate of 2D bounding box(float)
    - column 6: bbox_top (ymin), upper left pixel coordinate of 2D bounding box(float)
    - column 7: bbox_right (xmax), right pixel coordinate of 2D bounding box(float)
    - column 8: bbox_bottom (ymax), bottom pixel coordinate of 2D bounding box(float)
- 3D scale of the object
    - column 9: height of the object in real world(float)
    - column 10: width of the object in real world(float)
    - column 11: length of the object in real world(float)
- 3D physical location of object center in camera coordinate system
    - column 12: x coordinate of the object center in camera coordinates(float)
    - column 13: y coordinate of the object center in camera coordinates(float)
    - column 14: depth of the object center in camera coordinates(float)
- physical orientation
    - column 15: rotation angle of the object around the y-axis in camera coordinates(float, -pi to pi)
preprocessed label