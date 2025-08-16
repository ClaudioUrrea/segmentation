import cv2
import numpy as np
from mmcv.transforms import BaseTransform
from mmseg.registry import TRANSFORMS

@TRANSFORMS.register_module()
class GaussianBlur(BaseTransform):
    """Aplica Gaussian Blur a la imagen."""

    def __init__(self, kernel_size=(5, 5), sigma=0, prob=0.5):
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.prob = prob

    def transform(self, results):
        if np.random.rand() < self.prob:
            img = results['img']
            img = cv2.GaussianBlur(img, self.kernel_size, self.sigma)
            results['img'] = img
        return results
