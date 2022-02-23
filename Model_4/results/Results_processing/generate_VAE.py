from datetime import date
import fire as fire
import os
import sys

sys.path.append(os.path.join("..",".."))
sys.path.append(os.path.join("..","..","..",'Data_prep'))
sys.path.append(os.path.join("..","..","..",'DL_utils'))
sys.path.append(os.path.join("..","..",'..','ndvi'))

sys.path.append(os.path.join("..",".."))

import torch
from torch import nn
from Utils import *

from Net.VAE_v2 import GMVAE

from torchvision import transforms
from Custom_dataloader import *
from Transforms import phantom_segmentation
from Transforms import phantom_segmentation_
from Transforms import multi_image_resize
from Transforms import multi_ToTensor
from Transforms import output_transform
from Transforms import *



def main():
    res_dir=os.path.join("..",'GMVAE_A1_3')
    model_state=torch.load(os.path.join(res_dir,'best0.pt'))

    DB="/home/liiarpi-01/proyectopaltas/Local_data_base/Data_Base_v2"
    meta_dir="/home/liiarpi-01/proyectopaltas/Local_data_base/metadata_GMVAE_A1_3"

    model=GMVAE(image_dim=int(512),
        image_channels=1,
        repr_sizes=[3,6,12,24,48],
        layer_sizes=[200,100,50],
        w_latent_space_size=10,
        z_latent_space_size=10,
        y_latent_space_size=5,
        conv_kernel_size=7,
        conv_pooling=False,
        activators=[nn.Sigmoid(),nn.LeakyReLU(),nn.LeakyReLU(),nn.LeakyReLU(),nn.LeakyReLU()],
        conv_batch_norm=True,
        NN_batch_norm=True,
        stride=2,
        device="cpu")
    model.to("cuda")

    model.load_state_dict(model_state)

    d_tt=transforms.Compose([
        ndvi_desc(),
        multi_image_resize(ImType=['SenteraNDVI'],size=(512,512)),
        multi_ToTensor(ImType=['SenteraNDVI']),
        select_out_transform(selected=['SenteraNDVI','Place','Date','landmarks'])### ----------------------------------------
        ])

    datab=Dataset_direct(root_dir=DB,ImType=['SenteraRGB','SenteraNIR','SenteraMASK'],Intersec=False,transform=d_tt)
    print(len(datab))

    parallel_gen_metadata_from_GMVAE(data_base=datab,
                                    out_meta_dir=meta_dir,
                                    model=model,
                                    batch_size=20,
                                    num_workers=6,
                                    args=['SenteraNDVI'],
                                    device_in='cuda')

    
if __name__ == "__main__":
    fire.Fire(main)