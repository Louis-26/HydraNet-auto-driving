# step 1: Configure the environment
```bash
cd $(git rev-parse --show-toplevel)/cv-multitask-learning-project
conda create -n hydranet python=3.11 -y
conda activate hydranet
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

# step 2: Download Dataset
```bash
cd $(git rev-parse --show-toplevel)/cv-multitask-learning-project/scripts
bash raw_data_download.sh
```


# step 3: Download Pretrained Weights
```bash