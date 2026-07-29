# HydraNet Pipeline Setup

With `mobilenetv2` as the shared encoder, we used `Lightweight Refinenet` for semantic segmentation and depth estimation, `YOLOv8` for object detection, 

## step 1: Environment Setup
```bash
cd $(git rev-parse --show-toplevel)/framework
conda create -n hydranet python=3.11 -y
conda activate hydranet
pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
cd scripts
bash pretrained_weight_download.sh  
```

Notes:

- The repo uses PyTorch, OpenCV, Albumentations, Matplotlib, Pillow, and TQDM.
- The model code assumes the shared pretrained checkpoint is available at `checkpoints/ExpKITTI_joint.ckpt`.

## step 2: Download and split dataset 


download objection detection/segmentation/depth estimation dataset from KITTI
```bash
cd $(git rev-parse --show-toplevel)/cv-multitask-learning-project
bash scripts/data_prepare.sh
```

What it downloads:

- KITTI raw drives and calibration archives from the official KITTI raw-data bucket.
- The script loops through the raw sequence list and expands each archive in place.

After that, you can find out the structure of the KITTI data under
`framework/data/`.

More details regarding the selection of KITTI dataset has been illustrated [here](../overview/dataset.md)


### reference
- KITTI raw data
  - webpage: http://www.cvlibs.net/datasets/kitti/raw_data.php
  - dataset download link: http://www.cvlibs.net/download.php?file=data_stereo_flow.zip
- KITTI depth
  - webpage: https://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_prediction
  - dataset download link: https://s3.eu-central-1.amazonaws.com/avg-kitti/data_depth_annotated.zip
- KITTI object detection
  - webpage: https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2d
  - dataset download link: https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip
- KITTI semantic segmentation
  - webpage: https://www.cvlibs.net/datasets/kitti/eval_semseg.php?benchmark=semantics
  - dataset download link: https://s3.eu-central-1.amazonaws.com/avg-kitti/data_semantics.zip

## step 3: Dataset Preprocessing
- produce YOLOv8 compatible label format for object detection task 

```bash
cd $(git rev-parse --show-toplevel)/framework
python dataloaders/dataset_preprocess.py
```

- provide object detection bounding box visualization in addition to the original image 

```bash
python ./dataloaders/od_dataset_annotate.py --dataset_dir ./data/kitti_object/
```

## step 4: test with the dummy dataset

prepare dummy dataset as toy example for verification that the code logic is working

```bash
cd $(git rev-parse --show-toplevel)/framework
python scripts/prepare_dummy.py
```

In the following steps, with `XX` in {od, ss, de, multitask} for object detection, semantic segmentation, depth estimation, and multitask learning, respectively.

- train step:
```bash
cd $(git rev-parse --show-toplevel)/framework
python scripts/XX/train_XX.py 
```

- inference step:
```bash
cd $(git rev-parse --show-toplevel)/framework
python scripts/XX/inference_XX.py 
```

- evaluation step:
```bash
python scripts/XX/evaluate_XX.py 
```

if XX is `od` or `multitask`, 
use the following to verify the code functionality 
```bash
python scripts/XX/evaluate_XX.py --split train
```

## step 5: Train/Inference/Evaluate on KITTI dataset
Repeat the same steps as above, but replace the dataset path with the KITTI dataset path.

### Object Detection (OD) solely
- Train
```bash
cd $(git rev-parse --show-toplevel)/framework
python scripts/od/train_od.py \
    --data_root ./data/kitti_object/ \
    --epochs 100 \
    --batch_size 32 \
    --device cuda \
    --save_dir ./checkpoints/runs/official 

```

- Inference
```bash
cd $(git rev-parse --show-toplevel)/framework
python scripts/od/inference_od.py \
    --weights ./checkpoints/runs/official/best_detection_model.pth \
    --source ./data/kitti_object/test/images/ \
    --conf_thres 0.5 \
    --iou_thres 0.5 \
    --out_dir ./outputs/official/od  
    
```

- Evaluation
```bash
cd $(git rev-parse --show-toplevel)/framework
python scripts/od/evaluate_od.py \
    --weights ./checkpoints/runs/official/best_detection_model.pth \
    --data_root ./data/kitti_object/ \
    --conf_thres 0.5 \
    --iou_thres 0.5 
```

### Semantic Segmentation (SS) solely
- Train
```bash
cd $(git rev-parse --show-toplevel)/framework
python scripts/ss/train_ss.py \
    --data_root ./data/kitti_semantics/ \
    --epochs 100 \
    --batch_size 32 \
    --device cuda \
    --save_dir ./checkpoints/runs/official 
```

- Inference
```bash
cd $(git rev-parse --show-toplevel)/framework
python scripts/ss/inference_ss.py \
    --weights ./checkpoints/runs/official/best_segmentation_model.pth \
    --source ./data/kitti_semantics/test/images \
    --out_dir ./outputs/official/ss
```

- Evaluation
```bash
cd $(git rev-parse --show-toplevel)/framework
python scripts/ss/evaluate_ss.py \
    --pred_dir outputs/official/ss/semantic \
    --data_root data/kitti_semantics \
    --split test \
    --num_classes 7

```

### Depth Estimation (DE) solely
- Train
```bash
cd $(git rev-parse --show-toplevel)/framework
python scripts/de/train_de.py \
    --data_root ./data/kitti_depth/ \
    --epochs 100 \
    --batch_size 32 \
    --device cuda \
    --save_dir ./checkpoints/runs/official 
```

- Inference
```bash
cd $(git rev-parse --show-toplevel)/framework
python scripts/de/inference_de.py \
    --weights checkpoints/runs/official/best_depth_model.pth \
    --source_img data/kitti_depth/test/images \
    --source_gt data/kitti_depth/test/depth \
    --out_dir outputs/official/de
```

- Evaluation
```bash
cd $(git rev-parse --show-toplevel)/framework
python scripts/de/evaluate_de.py \
    --pred_dir outputs/official/de/predicted_depth \
    --data_root data/kitti_depth \
    --split test
```

### Multitask 
- Train
```bash
cd $(git rev-parse --show-toplevel)/framework
python scripts/multitask/train_multitask.py \
    --data_root ./data \
    --save_dir checkpoints/runs/official \
    --epochs 100 \
    --batch_size 64 \
    --lr 0.0002 \
    --lambda_od 1.0 \
    --lambda_ss 1.0 \
    --lambda_de 2.0 \
    --num_workers 4 \
    --print_freq 10
```

- Inference
```bash
cd $(git rev-parse --show-toplevel)/framework
python scripts/multitask/inference_multitask.py \
    --weights checkpoints/runs/official/best_multitask_model.pth \
    --data_root data \
    --split test \
    --out_dir outputs/official/multitask 

```

- Evaluation
```bash
cd $(git rev-parse --show-toplevel)/framework
python scripts/multitask/evaluate_multitask.py
```
