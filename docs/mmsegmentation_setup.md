MMSegmentation Setup Guide

This guide provides specific setup instructions for MMSegmentation within the Intelligent Mining Systems project.

Prerequisites



Python 3.8+

PyTorch 1.9.0+ with CUDA support

MMCV-full 1.3.13+



Installation Steps



Install MMCV

pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.9.0/index.html





Install MMSegmentation

git clone https://github.com/open-mmlab/mmsegmentation.git

cd mmsegmentation

pip install -r requirements.txt

python setup.py develop





Project-Specific Setup

cd ../mmsegmentation  # In the project repo

pip install -v -e .







Custom Modifications



Backbones: Custom implementations in mmseg/models/backbones/ for BiSeNetV1, MobileNetV2, ResNet, SegFormer, Twins.

Datasets: mmseg/datasets/automine1d.py for AutoMine dataset handling.

Transforms: Custom augmentations like CLAHE and Gaussian Blur in mmseg/datasets/transforms/.



Verification

import mmseg

print(mmseg.\_\_version\_\_)  # Should print >=0.28.0



from mmseg.apis import init\_segmentor

config = 'configs/ResNet/Resnet\_lr0.001\_Clahe\_photo.py'

model = init\_segmentor(config, device='cpu')

print("MMSegmentation setup successful!")



Troubleshooting



MMCV Version Mismatch: Ensure MMCV version matches PyTorch and CUDA.

Custom Module Errors: Verify editable install (-e) and restart kernel.



For full project installation, refer to installation.md.

