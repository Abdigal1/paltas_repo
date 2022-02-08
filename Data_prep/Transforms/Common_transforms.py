from skimage.transform import resize
import numpy as np
import torch
from torchvision import transforms

class rgb_normalize(object):
    """Crop randomly the image in a sample.

    Args:
        regular (bool): regular o non regular segmentation mask
    output:

    """

    def __init__(self, ImType=["PhantomRGB"]):
        assert isinstance(ImType, list)
        self.ImType=ImType

    def __call__(self, sample):
        #For Imtype
        for Type in self.ImType:
            # transform sample and save
            sample[Type]=sample[Type]/255


        return sample

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
            sample[Type]=resize(sample[Type],self.size)


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
            sample[Type]=(self.TT(sample[Type])).to(torch.float)
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

class pos_fly_transform(object):
    """Crop randomly the image in a sample.

    Args:
        regular (bool): regular o non regular segmentation mask
    output:

    """

    def __init__(self,fly_map={
                                '11_junio_1':       0,
                                '12_mayo_1':        1,
                                '12_mayo_2':        2,
                                '13_agosto_1':      3,        
                                '14_abril_1':       4,
                                '14_abril_2':       5,
                                '14_julio_1':       6,
                                '15_setiembre_1':   7,       
                                '16_junio_1':       8,
                                '16_junio_2':       9,
                                '19_agosto_1':      10,
                                '19_mayo_1':        11,       
                                '19_mayo_2':        12,
                                '23_julio_1':       13,
                                '23_julio_2':       14,
                                '23_junio_1':       15,       
                                '23_junio_2':       16,
                                '24_setiembre_1':   17,
                                '26_mayo_1':        18,
                                '26_mayo_2':        19,       
                                '28_abril_1':       20,
                                '28_abril_2':       21,
                                '29_marzo_1':       22,
                                '29_marzo_2':       23,       
                                '2_julio_1':        24,
                                '2_junio_1':        25,
                                '5_agosto_1':       26,
                                '7_mayo_1':         27,
                                '9_julio_1':        28
    }
    ):
        self.fly_map=fly_map

    def __call__(self, sample):
        #For Imtype
        Lat=float(sample['PhantomRGB_metadata']['GPSInfo']["GPSLatitude"][2])
        Lon=float(sample['PhantomRGB_metadata']['GPSInfo']["GPSLongitude"][2])
        Alt=float(sample['PhantomRGB_metadata']['GPSInfo']["GPSAltitude"])
        pos=np.array([Lat,Lon,Alt])
        date_ohe=np.zeros(29)
        date_ohe[self.fly_map[sample['Date']]]=1
        sample["Encoded_date"]=date_ohe
        sample["GPSposition"]=pos
        return sample