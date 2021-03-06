import fire as fire
import os
import sys

sys.path.append(os.path.join("..",".."))
sys.path.append(os.path.join("..","..","..",'Data_prep'))
sys.path.append(os.path.join("..","..","..",'DL_utils'))
sys.path.append(os.path.join("..","..",'..','ndvi'))

sys.path.append(os.path.join("..",".."))

import torch
from Utils import *

from Net.VAE_v2 import b_encodeco

import torch
from torch import nn
from Utils import *

from torchvision import transforms
from Custom_dataloader import *
from Transforms import phantom_segmentation
from Transforms import phantom_segmentation_
from Transforms import multi_image_resize
from Transforms import multi_ToTensor
from Transforms import output_transform
from Transforms import rgb_normalize



def main():
    res_dir=os.path.join("..",'VAE_v2_4')
    model_state=torch.load(os.path.join(res_dir,'best0.pt'))

    DB="/home/liiarpi-01/proyectopaltas/Local_data_base/Data_Base_v2"
    meta_dir="/run/user/1000/gvfs/afp-volume:host=MyCloudPR4100.local,user=aorus_1,volume=Paltas_DataBase/metadata_VAE_v4"

    model=b_encodeco(image_dim=int(200),
                 image_channels=3,
                 repr_sizes=[10,20,40],
                 layer_sizes=[],
                 latent_space_size=50,
                 conv_kernel_size=15,
                 conv_pooling=False,
                 conv_batch_norm=True,
                 NN_batch_norm=True,
                 stride=2,
                device='cpu')

    model.load_state_dict(model_state)

    d_tt=transforms.Compose([
        phantom_segmentation_(False),
        rgb_normalize(ImType=['PhantomRGB']),
        multi_image_resize(ImType=['PhantomRGB'],size=(200,200)),
        multi_ToTensor(ImType=['PhantomRGB']),
        select_out_transform(selected=['SenteraNDVI','Place','Date','landmarks'])### ----------------------------------------
        ])

    datab=Dataset_direct(root_dir=DB,ImType=['PhantomRGB'],Intersec=False,transform=d_tt)

    #gen_metadata_from_model(data_base=datab,
    #                        out_meta_dir=meta_dir,
    #                        model=model
    #                        )

    parallel_gen_metadata_from_VAE(data_base=datab,
                                    out_meta_dir=meta_dir,
                                    model=model,
                                    batch_size=20,
                                    num_workers=6,
                                    args=['SenteraNDVI'],
                                    device_in='cuda')

    
if __name__ == "__main__":
    fire.Fire(main)