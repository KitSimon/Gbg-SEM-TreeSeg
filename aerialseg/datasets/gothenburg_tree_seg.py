# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset

@DATASETS.register_module()
class GothenburgTreeSeg(CustomDataset):
    """Lantmäteriet ortofoto 2022.

    In segmentation map annotation for Potsdam dataset, 0 is the ignore index.
    ``reduce_zero_label`` should be set to True. The ``img_suffix`` and
    ``seg_map_suffix`` are both fixed to '.png'.
    """
    #CLASSES = ('impervious_surface', 'building', 'low_vegetation', 'tree',
    #           'car', 'clutter')

    #PALETTE = [[255, 255, 255], [0, 0, 255], [0, 255, 255], [0, 255, 0],
    #           [255, 255, 0], [255, 0, 0]]

    #CLASSES = 'trees'                                  #### kolla ifall bakgrund behöver klass, eller ifall "0-index" som ovan räcker ##  reduce_zero_label=True innebär 0 blir tolkas som 255, dvs "null")
                                                                            
    #PALETTE = [0, 255, 0]


    # CLASSES and PALETTE must have the same length, and that length must
    # equal num_classes in configs/aerialformer/aerialformer_tiny_512x512_gbg.py.
    #
    # Label encoding: AF_0_training_data.py burns class_value = CLASSES index + 1
    # (bg=1, water=2, tree=3).  reduce_zero_label=True shifts these to 0,1,2 at
    # load time and maps raw 0 to 255 ("unlabeled", excluded from loss/metrics),
    # so pixel value 0 must never be used for a real class.
    #
    # To add a class (e.g. 'building'): append its name here, append a color to
    # PALETTE, burn it in AF_0 with the next class_value, bump num_classes in
    # the model config, and retrain.
    CLASSES = ('bg', 'water', 'tree')

    PALETTE = [[255, 255, 255], [0, 0, 255], [0, 255, 0]]  # bg=white, water=blue, tree=green

    def __init__(self, **kwargs):
        kwargs.setdefault('img_suffix', '.png')
        kwargs.setdefault('seg_map_suffix', '.png')
        kwargs.setdefault('reduce_zero_label', True)
        super(GothenburgTreeSeg, self).__init__(**kwargs)

