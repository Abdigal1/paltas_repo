from os import stat
from utils import *
import matplotlib.pyplot as plt



class registration():
    """Aligne RGB and NIR image using RGB image as truth position
    
    Args:
        adjusted: Adjust of image color by factors indicated by manufacturer
        
    output:
        NDVI and RGB image
        if adjusted the data type is float

    """

    def __init__(self, adjusted=True):
        self.adjusted=adjusted

    def __call__(self, sample):
        rgb, nir, landmarks = sample['SenteraRGB'],\
            sample['SenteraNIR'], sample['landmarks']
        
        rgb, nir = registration(rgb, nir, self.adjusted)

        return{'SenteraRGB':rgb, 'SenteraNIR':nir , 'landmarks': landmarks}