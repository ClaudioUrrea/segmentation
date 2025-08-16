import numpy as np
from mmseg.datasets.transforms import CLAHE
from mmseg.registry import TRANSFORMS

@TRANSFORMS.register_module()
class RandomCLAHE(CLAHE):
    """CLAHE con probabilidad de aplicación."""

    def __init__(self, prob=0.5, **kwargs):
        super().__init__(**kwargs)
        self.prob = prob

    def transform(self, results):
        if np.random.rand() < self.prob:
            return super().transform(results)
        return results
