#from segmentation_methods import seg_mask
import numpy as np

from .segmentation_methods.segmentation_phantom import seg_mask

class phantom_segmentation(object):
    """Crop randomly the image in a sample.

    Args:
        regular (bool): regular o non regular segmentation mask
    output:

    """

    def __init__(self, regular=True):
        assert isinstance(regular, bool)
        self.regular=regular

    def __call__(self, sample):
        image, landmarks = sample['PhantomRGB'], sample['landmarks']

        mk,x_max,x_min,y_max,y_min=seg_mask(image,squared=self.regular)
        simg=(np.stack((mk,mk,mk),axis=2))*image

        return {'PhantomRGB': simg[x_min:x_max,y_min:y_max,:], 'landmarks': landmarks}