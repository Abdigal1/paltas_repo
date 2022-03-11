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

from Net.VAE_meta import b_encodeco

from torchvision import transforms
from Custom_dataloader import *
from Transforms import phantom_segmentation
from Transforms import phantom_segmentation_
from Transforms import multi_image_resize
from Transforms import multi_ToTensor
from Transforms import output_transform
from Transforms import *



def main():
    res_dir=os.path.join("..",'metaVAE_L_2')
    model_state=torch.load(os.path.join(res_dir,'best0.pt'))

    #DB="/run/user/1000/gvfs/afp-volume:host=MyCloudPR4100.local,user=aorus_1,volume=Paltas_DataBase/Data_Base_v2"
    DB="/home/lambda/paltas/Local_data_base/Data_Base_v2"
    meta_dir="/home/lambda/paltas/Local_data_base/metadata_metaVAE_L_2"

    device="cuda"

    d_tt=transforms.Compose([
        trans_registration(),
        multi_image_resize(ImType=['SenteraRGB','SenteraNIR'],size=(512,512)),
        stack_multiespectral(),
        pos_fly_transform(ImType=["SenteraRGB"]),
        concatenate_non_uniform_transform(),
        multi_ToTensor(ImType=['SenteraRGBNIR']),
        select_out_transform(selected=['SenteraRGBNIR','Non_uniform_input','Place','Date','landmarks'])### ----------------------------------------
        ])

    model=b_encodeco(image_dim=int(512),
                 image_channels=5,
                 non_uniform_dim=33,
                 pre_layer_sizes=[300,200],
                 pre_output=500,
                 repr_sizes=[8,16,32,64,128],
                 layer_sizes=[100,80,50],
                 latent_space_size=40,
                 conv_kernel_size=7,
                 activators=[nn.Sigmoid(),nn.LeakyReLU(),nn.LeakyReLU(),nn.LeakyReLU(),nn.LeakyReLU()],
                 conv_pooling=False,
                 conv_batch_norm=True,
                 NN_batch_norm=True,
                 stride=2,
                device="cuda",
                 Multi_GPU=True,
                 in_device="cuda:0")
                
    

    model.load_state_dict(model_state)

    model.to(device)

    datab=Dataset_direct(root_dir=DB,ImType=['SenteraRGB','SenteraNIR','SenteraMASK'],Intersec=False,transform=d_tt)

    #gen_metadata_from_model(data_base=datab,
    #                        out_meta_dir=meta_dir,
    #                        model=model
    #                        )

    parallel_gen_metadata_from_VAE(data_base=datab,
                                    out_meta_dir=meta_dir,
                                    model=model,
                                    batch_size=40,
                                    num_workers=9,
                                    args=['SenteraRGBNIR','Non_uniform_input'],
                                    device_in='cuda')

    
if __name__ == "__main__":
    fire.Fire(main)