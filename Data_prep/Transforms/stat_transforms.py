import numpy as np

from .Phantom_stats_trans.phantom_stats_methods import hist_stat_l
from .Phantom_stats_trans.phantom_stats_methods import hist_stat_h
from .Phantom_stats_trans.phantom_stats_methods import bl_perc

class hsv_stats_transfrom(object):
    """Crop randomly the image in a sample.

    Args:
        regular (bool): regular o non regular segmentation mask
    output:

    """

    def __init__(self):
        pass

    def __call__(self, sample):
        image, landmarks = sample['PhantomRGB'], sample['landmarks']

        m,st,mo=hist_stat_h(image)

        return {'stat_val': {'mean':m,'std':st,'mode':mo}, 'landmarks': landmarks}

class lab_stats_transfrom(object):
    """Crop randomly the image in a sample.

    Args:
        regular (bool): regular o non regular segmentation mask
    output:

    """

    def __init__(self):
        pass

    def __call__(self, sample):
        image, landmarks = sample['PhantomRGB'], sample['landmarks']

        m,st,mo=hist_stat_l(image)

        return {'stat_val': {'mean':m,'std':st,'mode':mo}, 'landmarks': landmarks}

class black_perc_transfrom(object):
    """Crop randomly the image in a sample.

    Args:
        regular (bool): regular o non regular segmentation mask
    output:

    """

    def __init__(self):
        pass

    def __call__(self, sample):
        image, landmarks = sample['PhantomRGB'], sample['landmarks']

        m=bl_perc(image)

        return {'bl_per': m, 'landmarks': landmarks}