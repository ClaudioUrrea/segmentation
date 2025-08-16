Training Guide

This guide provides instructions for training the BiSeNetV1 model on the AutoMine dataset.

Setup



Ensure the environment is set up as per installation.md.

Download and prepare the dataset as per dataset\_preparation.md.



Training



Select a configuration file from mmsegmentation/configs/.



Run the training script:

jupyter notebook scripts/For\_training.ipynb





Monitor training progress in the work\_dirs/ directory.





Optimal Configurations



MobileNetV2: Mobilenet\_lr0.001\_Clahe\_photo.py

ResNet-50: Resnet\_lr0.001\_Clahe\_photo.py

SegFormer-B0: Segformer\_lr0.0001\_Clahe\_photo.py

Twins-PCPVT-S: Twins\_lr0.01\_photo.py



