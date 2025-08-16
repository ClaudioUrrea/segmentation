Dataset Preparation Guide
This guide explains how to set up the AutoMine dataset for intelligent autonomous mining systems research, including both source domain (clean images) and target domains (visual degradations).
Table of Contents

Dataset Overview
Download Instructions
Directory Structure
Data Preprocessing
Validation
Custom Dataset Creation

Dataset Overview
The Intelligent Mining Systems dataset is based on the AutoMine dataset with custom annotations for semantic segmentation of traversable areas in autonomous mining environments.
Dataset Characteristics

Total Images: 120 images (100 training + 20 test base)
Source Domain: 100 clean images for training
Target Domains:
Lens Soiling: 10 original + 90 augmented = 100 test images
Sun Glare: 10 original + 90 augmented = 100 test images


Resolution: Variable (typically 1920×1080)
Format: RGB images (.jpg) + binary masks (.png)
Classes: 2 (background, traversable road)

Download Instructions
Option 1: Direct Download from FigShare

Access: https://doi.org/10.6084/m9.figshare.29897300
Download the complete dataset archive.

Option 2: Automated Download Script
python scripts/download_dataset.py --output_dir data/ --verify_checksums

Directory Structure
data/
├── automine1d/                     # Source domain (clean images)
│   ├── images/
│   │   ├── IMG_001.jpg
│   │   ├── IMG_002.jpg
│   │   └── ... (100 images)
│   ├── annotations/
│   │   ├── IMG_001.png
│   │   ├── IMG_002.png
│   │   └── ... (100 masks)
│   └── metadata/
│       ├── train_split.txt
│       ├── val_split.txt
│       └── image_info.csv
├── automine1d_distortion/          # Target domains
│   ├── lens_soiling/
│   │   ├── images/
│   │   │   ├── original/
│   │   │   └── augmented/
│   │   ├── annotations/
│   │   │   ├── original/
│   │   │   └── augmented/
│   │   └── metadata/
│   │       └── test_split.txt
│   └── sun_glare/
│       ├── images/
│       │   ├── original/
│       │   └── augmented/
│       ├── annotations/
│       │   ├── original/
│       │   └── augmented/
│       └── metadata/
│           └── test_split.txt
└── metadata/
    ├── dataset_statistics.csv
    └── augmentation_parameters.yaml

Data Preprocessing
import cv2
import numpy as np
from pathlib import Path

def validate_dataset(data_path):
    issues = []
    img_dir = Path(data_path) / 'images'
    mask_dir = Path(data_path) / 'annotations'
    
    img_files = set(f.name for f in img_dir.glob('*.jpg'))
    mask_files = set(f.name.replace('.png', '.jpg') for f in mask_dir.glob('*.png'))
    
    missing_images = mask_files - img_files
    missing_masks = img_files - mask_files
    
    if missing_images:
        issues.append(f"Missing images: {missing_images}")
    if missing_masks:
        issues.append(f"Missing masks: {missing_masks}")
    
    for mask_path in mask_dir.glob('*.png'):
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        unique_values = np.unique(mask)
        if not np.all(np.isin(unique_values, [0, 1, 255])):
            issues.append(f"Invalid mask values in {mask_path.name}: {unique_values}")
    
    return issues

Custom Dataset Creation

Image Collection
mkdir -p data/custom_sequence/images/


Annotation: Use Label Studio for binary mask creation.

Integration: Update dataset configuration and run validation checks.

