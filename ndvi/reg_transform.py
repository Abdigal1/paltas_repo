from os import stat
from utils import *
import matplotlib.pyplot as plt



class trans_registration():
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
        rgb_0 = (rgb[:,:,0]+0.243)/(255*1.62)
        rgb_1 = (rgb[:,:,1]+0.528)/(255*1.948)
        rgb_2 = (rgb[:,:,2]+0.144)/(255*1.294)
        nir_0 = (nir[:,:,0]+0.956)/(255*1.956)
        nir_1 = (nir[:,:,1]+0.341)/(255*2.767)
        nir_2 = nir[:,:,2]/255.0

        rgb_0[(rgb_0>1)|(rgb_0<0)]=0
        rgb_1[(rgb_1>1)|(rgb_1<0)]=0
        rgb_2[(rgb_2>1)|(rgb_2<0)]=0
        nir_0[(nir_0>1)|(nir_0<0)]=0
        nir_1[(nir_1>1)|(nir_1<0)]=0
        nir_2[(nir_2>1)|(nir_2<0)]=0
        rgb[:,:,0], rgb[:,:,1], rgb[:,:,2] = rgb_0, rgb_1, rgb_2
        nir[:,:,0], nir[:,:,1], nir[:,:,2] = nir_0, nir_1, nir_2
        
        sample['SenteraNIR']=nir
        sample['SenteraRGB']=rgb

        return sample