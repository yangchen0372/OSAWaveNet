# OSAWaveNet

Official implementation of **OSAWaveNet** for remote-sensing change detection on **LEVIR-CD**, **SYSU-CD**, and **HRCUS-CD**.


## Requirements
- Python ≥ 3.8
- PyTorch (CUDA optional)
- `numpy`, `opencv-python`, `tqdm` (and other common deps)

## Dataset Preparation
Set the dataset root in the corresponding config inside `Dataset_Helper/<Dataset_Name>_CFG.py`.

Dataset_Helper exposes unified loaders for LEVIR-CD, SYSU-CD, and HRCUS-CD that read data_root from these files.

## Training
Run the script that matches your dataset:
#### LEVIR-CD
python OSAWaveNet_LEVIR256_Train.py
#### SYSU-CD
python OSAWaveNet_SYSU256_Train.py
#### HRCUS-CD
python OSAWaveNet_HRCUS256_Train.py

## Inference & Validation

python OSAWaveNet_Inference.py 

Download pretrained checkpoints here: https://drive.google.com/drive/folders/1pD2nDBjG80WuUwg3_pNATRpcLgE3s1oF?usp=sharing

