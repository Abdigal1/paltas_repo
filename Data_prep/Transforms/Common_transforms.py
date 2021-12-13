from skimage.transform import resize
import numpy as np
import torch
from torchvision import transforms

class multi_image_resize(object):
    """Crop randomly the image in a sample.

    Args:
        regular (bool): regular o non regular segmentation mask
    output:

    """

    def __init__(self, ImType=["Phantom"],size=(2048,2048)):
        assert isinstance(ImType, list)
        assert isinstance(size, tuple)
        self.ImType=ImType
        self.size=size

    def __call__(self, sample):
        #For Imtype
        for Type in self.ImType:
            # transform sample and save
            sample[Type]=resize(sample[Type],self.size).astype('uint8')


        return sample

class multi_ToTensor(object):
    """Crop randomly the image in a sample.

    Args:
        regular (bool): regular o non regular segmentation mask
    output:

    """

    def __init__(self, ImType=["Phantom"]):
        assert isinstance(ImType, list)
        self.ImType=ImType
        self.TT=transforms.ToTensor()

    def __call__(self, sample):
        #For Imtype
        for Type in self.ImType:
            # transform sample and save
            sample[Type]=self.TT(sample[Type])
        return sample

class output_transform(object):
    """Crop randomly the image in a sample.

    Args:
        regular (bool): regular o non regular segmentation mask
    output:

    """

    def __init__(self, output_map={
                                'Control':1,
                                'H50%':2,
                                'H75%':3,
                                'K_Control':4,
                                'K_Deficiencia':5,
                                'K_Exceso':6,
                                'N_Control':7,
                                'N_Deficiencia':8,
                                'N_Exceso':9,
                                'P_Control':10,
                                'P_Deficiencia':11,
                                'P_Exceso':12
                                    }
                                    ):
        assert isinstance(output_map, dict)
        self.output_map=output_map

    def __call__(self, sample):
        #For Imtype
        sample['landmarks']=self.output_map[sample['landmarks']]
        return sample