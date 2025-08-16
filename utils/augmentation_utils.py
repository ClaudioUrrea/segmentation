# augmentation_utils.py
import cv2
import numpy as np
import albumentations as A

def apply_photometric_distortion(image, brightness_delta=32, contrast_range=(0.5, 1.5)):
    """Apply photometric distortion."""
    transform = A.Compose([
        A.RandomBrightnessContrast(brightness_limit=(-brightness_delta/255, brightness_delta/255),
                                   contrast_limit=contrast_range, p=1.0),
    ])
    augmented = transform(image=image)
    return augmented['image']

def apply_clahe(image, clip_limit=4.0, tile_grid_size=(8, 8)):
    """Apply CLAHE augmentation."""
    if len(image.shape) == 3:
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    else:
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        return clahe.apply(image)

def combined_augmentation(image):
    """Combined PD + CLAHE."""
    image = apply_photometric_distortion(image)
    image = apply_clahe(image)
    return image