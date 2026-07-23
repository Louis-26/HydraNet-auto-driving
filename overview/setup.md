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