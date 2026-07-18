# CV Multi-Task Learning Project

This project implements a **multi-task perception model** inspired by HydraNet for autonomous driving, performing:
- Semantic Segmentation
- Monocular Depth Estimation
- Object Detection

---

## Repository Structure

>  **Note:**  
> This GitHub repository is still a work in progress. Over the course of the semester, we experimented with different approaches and ideas, which led to several folders and scripts being added over time.  
> We plan to clean up, reorganize, and document everything in a more readable and modular structure in the near future.

### For now, please refer to the `final-submission` branch.

We have created a **separate branch** named `final-submission` for evaluation and report purposes. This branch contains only the two clean, final directories used in our write-up and presentation:

- `yolo_seg_depth/` – Final multitask implementation with YOLOv8 detection head  
- `ssd_seg_depth/` – Final multitask implementation with SSD detection head  

All other directories in the `main` branch (e.g., legacy scripts, experimental trials) can be **ignored for submission**.  
They may be useful for future improvements or extensions of the project.

> We will continue to update and refine this repository going forward.



## Run Inference Locally

### Step 1: Clone the Repo

```bash
git clone https://github.com/rrohan2/cv-multitask-learning-project.git
```

> **Note:** Before proceeding to the next step, please ensure that [Miniconda or Anaconda](https://www.anaconda.com/download/success) is installed on your system.  
> You’ll need the `conda` command available to create and manage the environment.  
> You can check if conda is already installed by running:
> ```bash
> conda --version
> ```
> If this command fails, follow the [Anaconda installation guide](https://www.anaconda.com/docs/getting-started/anaconda/install) or [Miniconda installation guide](https://www.anaconda.com/docs/getting-started/miniconda/install) to set it up.  

### Step 2: Create the Environment

Go to your IDE and open a terminal.

> For example, in **VS Code**, you can open the terminal using:
> - Shortcut: **Ctrl + Shift +** <kbd>`</kbd> (backtick key)
> - Or:  **Terminal** → **New Terminal**

Then run the following commands in IDE's terminal to create and activate the environment:

```bash
cd cv-multitask-learning-project
conda create --name cv-project python=3.8 pytorch torchvision torchaudio pytorch-cuda cudatoolkit -c pytorch -c nvidia -c conda-forge -y
conda activate cv-project
pip install -r requirements.txt
```
### Step 3: Setup Inference

The `checkpoints/` directory contains pretrained weights for the HydraNet model, trained for:
- Semantic Segmentation
- Monocular Depth Estimation

The `data/` directory includes an example video, broken down into individual `.png` image frames.  
This sample is provided **only for inference** (example output), to visualize the model's output on a small video snippet segregated in screenshots.  
**It is not training data.**

> ⚠️ **Note:** The full training dataset is not included in this repository.  
> It can either be downloaded manually using instructions in the [Appendix](#appendix) at the bottom of this page,  
> or we can set up a Google Drive folder with the training data.

If you want to test the model on your own video:
- Extract the frames as `.png` images
- Replace the contents of the `data/` folder with your own frames
- Ensure all images are in `.png` format and have consistent resolution

You can run the inference script with the existing data as of now to generate the output using the following command in the IDE terminal.

```bash
python scripts/inference.py
```

Once the script is executed, an output video will be generated showing:

- Original video frames on top  
- Semantic segmentation in the middle  
- Depth estimation on the bottom

The video will be saved at: `outputs/videos/out.mp4`

You can open this file using any media player of your choice (e.g., VLC, MPV, or your system’s default player).



## Project Structure 

Please use the directory and file structure/naming convention as shown below to maintain consistency throughout the project.

> 📝 **Note:** Some directories may not be visible on GitHub if they are empty. Git does not track empty folders by default.  
> However, you should still use and maintain this structure for consistency and future development.

- `cv-multitask-learning-project/`
  - `multitask_project/` – Core model code and task heads
    - `encoder.py` – MobileNetV2 encoder
    - `decoder.py` – Lightweight RefineNet decoder
    - `multitask_model.py` – Main model integrating encoder, decoder, and heads
    - `utils.py` – Utility functions for preprocessing and visualization
    - `__init__.py`
    - `heads/` – Task-specific prediction heads
      - `ssd_head.py` – SSD detection head
      - `yolov8_head.py` – YOLOv8 detection head
      - `detection_utils.py` – Shared helper functions for detection (e.g., NMS, anchors)
      - `__init__.py`
  - `scripts/` – Inference and training entry points
    - `inference.py` – Run segmentation + depth inference on images
    - `evaluate.py` – Evaluation logic (to be implemented)
    - `train_multitask.py` – Train segmentation + depth [+ normals]
    - `train_seg_depth.py` – Train segmentation + depth only
    - `train_detection.py` – Train SSD or YOLOv8 detection heads
  - `checkpoints/` – Pretrained model weights for inference (e.g., `ExpKITTI_joint.ckpt`)
  - `data/` – Sample inference data (video frames and color map `cmap_kitti.npy`)
  - `kitti_rawdata/` – Training Data (Local System)
  - `outputs/` – Generated outputs
    - `logs/` – Training logs, TensorBoard runs
    - `predictions/` – Optional saved outputs
    - `videos/` – Final output videos
      - `out.mp4` – Stacked result video
  - `notebooks/` – Jupyter notebooks for quick experiments
  - `requirements.txt` – Python dependencies
  - `README.md` – Project documentation (this file)



> **Note on Scripts and Files:**  
> Most of the scripts and modules are placeholders as of now and will be implemented in the coming weeks.  
> Some of them might not end up being used, but I have created in advance to keep the project structure clean, modular, and easy to expand as needed.
---

---

## Appendix

### Downloading Full Training Data

I’m still deciding on the most elegant way to manage and share the full training dataset,  
but for now, the workflow is planned as follows:

1. **Create a directory named `kitti_rawdata/`** inside the `cv-multitask-learning-project/` root folder.

> This folder is already included in `.gitignore`, so it will **not be pushed to GitHub**.

2. **Download the zip file** containing the data downloader script ([zip file for downloader script](https://drive.google.com/file/d/1I2vAyBpTQCSCvkpjU8uytluSOLWZ4Rh3/view?usp=drive_link)).

3. **Extract the zip inside the `kitti_rawdata/` folder.**

4. **Run the download script** from your terminal:

```bash
cd cv-multitask-learning-project/kitti_rawdata
./raw_data_downloader.sh
```

This will begin downloading the full KITTI raw dataset into the `kitti_rawdata/` folder.

> ⏳ **Note:** Downloading might take **2–3 hours or more** (not sure — I left it overnight).  
> Please start this process **as soon as possible** so the data is ready for all of us when needed.  
> I'm also **not entirely sure if this exact raw dataset will be used for training**   
> we might later switch to some other preprocessed format or subset.  
> But we need to start somewhere, and this is a good starting point for now.


## For Midpoint Check-In

For the midpoint check-in, I was planning to demonstrate the following:

- We have successfully implemented **semantic segmentation** and **depth estimation** tasks using the HydraNet-based architecture.
- We can include a result snapshot(like below) and **result table** showing performance metrics (e.g., mIoU for segmentation, RMSE for depth) on the example inference dataset.

![alt text](<Screenshot from 2025-03-31 04-14-37.png>)


### In Progress
- We are currently working on integrating **object detection heads** using:
  - SSD (Single Shot MultiBox Detector)
  - YOLOv8

We will explore and finalize the evaluation format for these detection tasks soon possibly using mAP (mean Average Precision), precision/recall etc.

### Future Work

- In future work we can write that we can use pointpillar to use lidar data for object detection.
- We can extend the model to Cityscape or other datasets.
