MMSegmentation Setup for Intelligent Mining Systems

This document provides specific instructions for setting up and using MMSegmentation within the Intelligent Systems for Autonomous Mining Operations project.

Installation



Navigate to MMSegmentation directory:

cd mmsegmentation





Install in development mode:

pip install -v -e .





Verify installation:

import mmseg

print(f"MMSeg version: {mmseg.\_\_version\_\_}")







Configuration Files

The configs/ directory contains configuration files for different backbones and hyperparameters:



MobileNet/: Configurations for MobileNetV2 backbone.

ResNet/: Configurations for ResNet-50 backbone.

SegFormer/: Configurations for SegFormer-B0 backbone.

Twins/: Configurations for Twins-PCPVT-S backbone.



Each directory includes configurations for different learning rates (0.0001, 0.001, 0.01) and augmentation strategies (CLAHE, Photometric Distortion, etc.). Optimal configurations are:



MobileNet: Mobilenet\_lr0.001\_Clahe\_photo.py

ResNet: Resnet\_lr0.001\_Clahe\_photo.py

SegFormer: Segformer\_lr0.0001\_Clahe\_photo.py

Twins: Twins\_lr0.01\_photo.py



Custom Backbones

The mmseg/models/backbones/ directory contains custom implementations:



bisenetv1.py: BiSeNetV1 architecture tailored for mining environments.

mobilenet\_v2.py, resnet.py, mit.py, twins.py: Modified backbones for single-domain generalization.



Custom Dataset

The mmseg/datasets/automine1d.py defines the AutoMine dataset with support for clean and degraded (lens soiling, sun glare) images.

Custom Transforms

The mmseg/datasets/transforms/ directory includes:



gaussian\_blur.py: Gaussian blur augmentation.

random\_clahe.py: CLAHE augmentation.

transforms.py: Main transformation pipeline.



Usage

Refer to scripts/For\_training.ipynb and scripts/For\_testing.ipynb for training and evaluation instructions.

