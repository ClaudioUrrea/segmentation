# data_preprocessing.py
import cv2
import numpy as np
from pathlib import Path

def preprocess_image(img_path, target_size=(1920, 1080)):
    """Preprocess an image for inference."""
    img = cv2.imread(img_path)
    img = cv2.resize(img, target_size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def preprocess_mask(mask_path, target_size=(1920, 1080)):
    """Preprocess a mask for evaluation."""
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)
    mask = (mask > 127).astype(np.uint8)
    return mask