# 同济子豪兄 2023-6-25
from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset

@DATASETS.register_module()
class automine1d(BaseSegDataset):

    METAINFO = {
        'classes':['background', 'road'],
        'palette':[[0,0,0], [0,128,0]]
    }
    def __init__(self, **kwargs):
        super().__init__(img_suffix='.png', seg_map_suffix='.png',
                    **kwargs)
#        assert osp.exists(self.img_dir)