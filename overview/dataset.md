# KITTI dataset
## Dataset structure
data/ 
├── kitti_object/
|   ├── train/
|   │   ├── images/      
|   │   ├── labels/      
│   |   └── YOLO_labels/
|   ├── val/   
|   │   ├── images/      
|   │   ├── labels/      
│   |   └── YOLO_labels/
│	├── test/
|   │   ├── images/      
|   │   ├── labels/      
│   |   └── YOLO_labels/
├── kitti_semantics
|   ├── train/
|   │   ├── images/      
│   |   ├── instance/       
│   |   ├── semantic/       
│   |   └── semantic_rgb/       
|   ├── val/
|   │   ├── images/      
│   |   ├── instance/       
│   |   ├── semantic/       
│   |   └── semantic_rgb/       
|   ├── test/
|   │   ├── images/      
│   |   ├── instance/       
│   |   ├── semantic/       
│   |   └── semantic_rgb/       
├── kitti_depth/
|   ├── train/
|   └── val/ 


## Dataset overview
In each of the subset below, we only utilize image_2 as left camera colorful image set and rename that to `images`, and the corresponding label data folder was renamed as following:
- KITTI depth: `depth`
- KITTI object detection: `labels`
- KITTI semantic segmentation: `semantic`

## KITTI Depth Estimation
As the number of images is huge in original dataset, we include only part of original dataset from original depth & raw data(so that it is scalable)     
"2011_09_26_drive_0001"   # City
"2011_09_26_drive_0019"   # Residential
"2011_09_26_drive_0015"   # Road
"2011_09_28_drive_0016"   # Campus
"2011_09_28_drive_0053"   # Person


## KITTI Segmentation
We include all available images and semantic/instance segmented images on KITTI.

For semantic segmented images, it includes **34** classes and i-th class has intensity value i. In reference dataset, each segmented image is relatively dark as the intensity value ranges from 0 to 33.  


## KITTI Object Detection
We include all available images and labels for each folder on KITTI.

ground truth label data type is the following(15 columns): 
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

example of label data:
```txt
Pedestrian 0.00 0 -0.20 712.40 143.00 810.73 307.92 1.89 0.48 1.20 1.84 1.47 8.41 0.01
```

However, the only useful attributes for our object detection task is 
`(class_id, center_x, center_y, width, height)` of the bounding box. Therefore, we convert the original label data to a simplified format with 5 columns in data preprocesing:
- column 1: class_id (integer, 0-13, 14 classes in total)
- column 2: center_x (float, normalized by image width)
- column 3: center_y (float, normalized by image height)
- column 4: width (float, normalized by image width)    
- column 5: height (float, normalized by image height)