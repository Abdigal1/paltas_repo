import sys
import os

sys.path.append(os.path.join("..","Data_prep"))
sys.path.append(os.path.join("..","DL_utils"))

from Train_utils.TT_class import trainer

import pathlib
import fire as fire
from Net.VAE_meta import b_encodeco
from torchvision import transforms
from Custom_dataloader import *
from Transforms import phantom_segmentation
from Transforms import multi_image_resize
from Transforms import multi_ToTensor
from Transforms import hue_transform
from Transforms import pos_fly_transform
from Transforms import concatenate_non_uniform_transform
from Transforms import multi_ToTensor
from Transforms import only_tensor_transform

from torch import nn

def main():
#    DB="/run/user/1000/gvfs/afp-volume:host=MyCloudPR4100.local,user=aorus_1,volume=Paltas_DataBase/Data_Base_v2"
    DB="/home/liarpi2/proyectopaltas/Local_data_base/Data_Base_v2"
    #DB="//MYCLOUDPR4100/Paltas_DataBase/Data_Base_v2"

    device='cuda'

    d_tt=transforms.Compose([
        phantom_segmentation(False,non_uniform_input=True),
        #rgb_normalize(ImType=['PhantomRGB']),
        hue_transform(),
        multi_image_resize(ImType=['PhantomRGB'],size=(200,200)),
        pos_fly_transform(),
        concatenate_non_uniform_transform(),
        multi_ToTensor(ImType=['PhantomRGB']),
        only_tensor_transform(),
        ])

    model=b_encodeco(
                image_dim=int(200),
                 image_channels=1,
                 non_uniform_dim=30,
                 repr_sizes=[2,4,8,16],
                 pre_layer_sizes=[300,200],
                 layer_sizes=[200,100],
                 pre_output=500,
                 latent_space_size=10,
                 conv_kernel_size=5,
                 activators=[nn.Sigmoid(),nn.LeakyReLU(),nn.LeakyReLU(),nn.LeakyReLU()],
                 conv_pooling=False,
                 conv_batch_norm=True,
                 NN_batch_norm=True,
                 stride=2,
                device=device)
    model.to(device)
    print("model loaded")

    datab=Dataset_direct(root_dir=DB,ImType=['PhantomRGB'],Intersec=False,transform=d_tt)
    print("data loaded")

    #os.path.join("..","Data_prep")
    T_ID="Meta_VAE_A2_1"
    pth=os.path.join(str(pathlib.Path().absolute()),"results",T_ID)
    print(pth)



    tr=trainer(
        model=model,
        dataset=datab,
        epochs=20,
        folds=2,
        batch_size=5,
        use_cuda=True,
        loss_list=['KLD','reconstruction',"total_loss"],
        data_dir=pth,
        in_device=None,
        num_workers=6,
        uniform=False,
        args=["PhantomRGB","Non_uniform_input"]
    )

    tr.K_fold_train()
    
if __name__ == "__main__":
    fire.Fire(main)