import sys
import os

sys.path.append(os.path.join("..","ndvi"))

from ndvi_transform import ndvi_desc

from skimage.transform import resize
from skimage.color import rgb2hsv
import numpy as np
import torch
from torchvision import transforms

class hue_transform(object):
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
            #Transform to HSV and save H
            sample[Type]=rgb2hsv(sample[Type])[:,:,0]
        return sample

class stack_multiespectral(object):
    """Crop randomly the image in a sample.

    Args:
        regular (bool): regular o non regular segmentation mask
    output:

    """

    def __init__(self, ImType=["SenteraRGB",'SenteraNIR']):
        assert isinstance(ImType, list)
        #if self.ImType!=["SenteraRGB",'SenteraNIR']:
        self.ImType=ImType

    def __call__(self, sample):
        sample['SenteraRGBNIR']=np.concatenate((
            sample["SenteraRGB"],
            sample['SenteraNIR'][:,:,np.array([0,2])]
        ),axis=2)
        return sample


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

    def __init__(self,ImType=["PhantomRGB"],only_decimals=True):
        self.ImType=ImType
        self.only_decimals=only_decimals
        if self.ImType[0]=='PhantomRGB':
            self.fly_map={'11_junio_1':    0,'12_mayo_1':   1,'12_mayo_2':   2,
                            '13_agosto_1': 3,'14_abril_1':  4,'14_abril_2':  5,
                            '14_julio_1':  6,'15_setiembre_1': 7,'16_junio_1':  8,
                            '16_junio_2':  9,'19_agosto_1': 10,'19_mayo_1':   11,       
                            '19_mayo_2':   12,'23_julio_1':  13,'23_julio_2':  14,
                            '23_junio_1':  15,'23_junio_2':  16,'24_setiembre_1':17,
                            '26_mayo_1':   18,'26_mayo_2':   19,'28_abril_1':  20,
                            '28_abril_2':  21,'29_marzo_1':  22,'29_marzo_2':  23,       
                            '2_julio_1':   24,'2_junio_1':   25,'5_agosto_1':  26,
                            '7_mayo_1':    27,'9_julio_1':   28}
        elif self.ImType[0]=='SenteraRGB':
            self.fly_map={'11_junio_1':    0,'12_mayo_1':   1,'12_mayo_2':   2,
                            '13_agosto_1': 3,'13_agosto_2': 4,'14_abril_1':  5,'14_abril_2':  6,
                            '14_julio_1':  7,'15_setiembre_1': 8,'16_junio_1':  9,
                            '16_junio_2':  10,'19_agosto_1': 11,'19_mayo_1':   12,       
                            '19_mayo_2':   13,'23_julio_1':  14,'23_julio_2':  15,
                            '23_junio_1':  16,'23_junio_2':  17,'24_setiembre_1':18,
                            '26_mayo_1':   19,'26_mayo_2':   20,'28_abril_1':  21,
                            '28_abril_2':  22,'29_marzo_1':  23,'29_marzo_2':  24,
                            '2_julio_1':   25,'2_junio_1':   26,'5_agosto_1':  27,
                            '5_agosto_2':  28,'7_mayo_1':    29,'9_julio_1':   30,
                            '9_julio_2':31}

    def __call__(self, sample):
        #TODO: For Imtype
        Lat=float(sample[self.ImType[0]+'_metadata']['GPSInfo']["GPSLatitude"][2])
        Lon=float(sample[self.ImType[0]+'_metadata']['GPSInfo']["GPSLongitude"][2])
        Alt=float(sample[self.ImType[0]+'_metadata']['GPSInfo']["GPSAltitude"])
        if self.only_decimals:
            Lat=Lat-int(Lat)
            Lon=Lon-int(Lon)
            Alt=Alt-int(Alt)
        #pos=np.array([Lat,Lon,Alt])
        pos=np.array([Alt])
        if self.ImType[0]=='PhantomRGB':
            date_ohe=np.zeros(29)
        elif self.ImType[0]=='SenteraRGB':
            date_ohe=np.zeros(32)
        date_ohe[self.fly_map[sample['Date']]]=1
        sample["Encoded_date"]=date_ohe
        sample["GPSposition"]=pos
        return sample

class concatenate_non_uniform_transform(object):
    """Crop randomly the image in a sample.

    Args:
        regular (bool): regular o non regular segmentation mask
    output:

    """

    def __init__(self,key_to_concatenate=["GPSposition","Encoded_date"]):
        self.key_to_concatenate=key_to_concatenate
        self.TT=transforms.ToTensor()

    def __call__(self, sample):
        #TODO: For Imtype
        sample["Non_uniform_input"]=(self.TT(
                                            np.hstack([sample[k] for k in self.key_to_concatenate]).reshape(1,-1)
                                            #np.hstack([sample[k] for k in self.key_to_concatenate])
                                            )).to(torch.float)
        return sample

class only_tensor_transform(object):
    #def __init__(self):
    def __call__(self, sample):
        for k in list(sample.keys()):
            if not isinstance(sample[k],torch.FloatTensor):
                sample.pop(k)
        return sample


