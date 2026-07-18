# HydraNet Pipeline Setup

With `mobilenetv2` as the shared encoder, we used `Lightweight Refinenet` for semantic segmentation and depth estimation, `YOLOv8` for object detection, 

## step 1: Environment Setup

The project now includes a bootstrap script:

`cv-multitask-learning-project/scripts/setup_env.sh`

Recommended usage:

```bash
cd cv-multitask-learning-project/scripts
bash setup_env.sh hydranet
```

What the script does:

```bash
cd $(git rev-parse --show-toplevel)/cv-multitask-learning-project
conda create -n hydranet python=3.11 -y
conda activate hydranet
pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
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
`cv-multitask-learning-project/data/` with


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

## step 3: test with the demo dataset

## toy train/infer/eval
prepare toy example
```bash
cd $(git rev-parse --show-toplevel)/cv-multitask-learning-project
mkdir -p dummy_data/labels dummy_data/images
cp kitti_raw_data/2011_09_26/2011_09_26_drive_0001_sync/image_02/data/0000000000.png dummy_data/images/
cp kitti_raw_data/2011_09_26/2011_09_26_drive_0001_sync/image_02/data/0000000001.png dummy_data/images/
echo "Car 0.0 0 0.0 100.0 100.0 200.0 200.0 1.5 1.6 3.1 0.0 1.6 5.0 0.0" > dummy_data/labels/0000000000.txt
echo "Car 0.0 0 0.0 150.0 120.0 250.0 220.0 1.5 1.6 3.1 0.0 1.6 5.0 0.0" > dummy_data/labels/0000000001.txt
echo "cv-multitask-learning-project/dummy_data" >> ../.gitignore
```
### train
```bash
python scripts/train_detection.py \
    --train-image-dir dummy_data/images \
    --train-label-dir dummy_data/labels \
    --epochs 1 \
    --batch-size 2 \
    --output-dir ./dummy_data/outputs \
    --amp
```

### inference
```bash
python scripts/inference.py \
  --input dummy_data/images \
  --seg-ckpt checkpoints/ExpKITTI_joint.ckpt \
  --det-ckpt ./dummy_data/outputs/best.pth
```

### evaluation
```bash
python scripts/evaluate.py \
  --image-dir dummy_data/images \
  --label-dir dummy_data/labels \
  --seg-ckpt checkpoints/ExpKITTI_joint.ckpt \
  --det-ckpt ./dummy_data/outputs/best.pth
```

## step 4: Training 
Freeze the MobileNetV2 encoder and train the YOLO detection head only. The training script is:

```bash
python scripts/train_detection.py \
  --train-image-dir /path/to/images/train \
  --train-label-dir /path/to/labels/train \
  --val-image-dir /path/to/images/val \
  --val-label-dir /path/to/labels/val \
  --seg-ckpt checkpoints/ExpKITTI_joint.ckpt \
  --det-num-classes 14 \
  --parser kitti
```

What this does:

- Loads the shared pretrained seg/depth checkpoint.
- Attaches the YOLOv8-style detection head.
- Freezes the shared encoder/decoder by default.
- Trains only the detection head unless `--train-backbone` is enabled.
- Saves checkpoints under `outputs/yolo_seg_depth/experiments/hydranet_od`.

Training checkpoint output:

- `best.pth`
- `epoch_XXX.pth`

The training script uses:

- YOLOv8-style detection loss with DFL + CIoU + BCE.
- Albumentations-based resizing and augmentation.
- AMP when `--amp` is enabled.

## step 5: Inference

The main inference entry point is:

```bash
python scripts/inference.py \
  --input data \
  --seg-ckpt checkpoints/ExpKITTI_joint.ckpt \
  --det-ckpt outputs/yolo_seg_depth/experiments/hydranet_od/best.pth
```

What it does:

- Loads the shared model and YOLO detection head.
- Runs segmentation, depth, and YOLO detection on each frame.
- Stacks the outputs into a single video.
- Saves the result to `outputs/videos/out.mp4`.

Optional behavior:

- Use `--save-frames` to also export per-frame images.
- Use `--conf-thres` and `--iou-thres` to tune detection filtering.

## step 6: Evaluation 

The evaluation entry point is:

```bash
python scripts/evaluate.py \
  --image-dir /path/to/images/val \
  --label-dir /path/to/labels/val \
  --seg-ckpt checkpoints/ExpKITTI_joint.ckpt \
  --det-ckpt outputs/yolo_seg_depth/experiments/hydranet_od/best.pth
```

The evaluator reports:

- `mAP50`
- per-class `AP50`

The output can optionally be dumped to JSON with `--output-json`.

## 7) Compatibility Scripts

These scripts remain in the repo as compatibility or helper entry points:

- `cv-multitask-learning-project/scripts/train_multitask.py`
- `cv-multitask-learning-project/scripts/train_seg_depth.py`

In this cleaned branch, the practical emphasis is the YOLO detection path, so these scripts are kept as lightweight wrappers or stage-1 helpers rather than the primary workflow.

## 8) What Changed

### New code files created

- `cv-multitask-learning-project/multitask_project/heads/bbox_loss.py`
- `cv-multitask-learning-project/multitask_project/heads/tal.py`
- `cv-multitask-learning-project/multitask_project/heads/detection_loss.py`
- `cv-multitask-learning-project/scripts/yolo_pipeline.py`
- `cv-multitask-learning-project/scripts/setup_env.sh`

### Code files changed

- `cv-multitask-learning-project/multitask_project/multitask_model.py`
- `cv-multitask-learning-project/multitask_project/heads/__init__.py`
- `cv-multitask-learning-project/multitask_project/heads/detection_utils.py`
- `cv-multitask-learning-project/multitask_project/heads/yolov8_head.py`
- `cv-multitask-learning-project/scripts/raw_data_downloader.sh`
- `cv-multitask-learning-project/scripts/train_detection.py`
- `cv-multitask-learning-project/scripts/train_multitask.py`
- `cv-multitask-learning-project/scripts/train_seg_depth.py`
- `cv-multitask-learning-project/scripts/inference.py`
- `cv-multitask-learning-project/scripts/evaluate.py`
- `overview/setup.md`

### Changed or created `.sh` files

- Created: `cv-multitask-learning-project/scripts/setup_env.sh`
- Changed: `cv-multitask-learning-project/scripts/raw_data_downloader.sh`
