import pathlib
import fire as fire
from B_VAE.VAE_v2 import GMVAE
from Train_utils import train_utils
from Train_utils.TT_class import trainer
from B_VAE.Utils_imp_VAE import loss_fn_b
from Train_utils.train_utils import train,test,K_fold_train
from torch import nn

import sys
import os
sys.path.append(os.path.join("..","Data_prep"))
from torchvision import transforms
from Custom_dataloader import *
from Transforms import phantom_segmentation
from Transforms import multi_image_resize
from Transforms import multi_ToTensor
from Transforms import output_transform
from Transforms import rgb_normalize

import torch

def main():
    #DB="/run/user/1000/gvfs/afp-volume:host=MyCloudPR4100.local,user=aorus_1,volume=Paltas_DataBase/Data_Base_v2"
    #DB="//MYCLOUDPR4100/Paltas_DataBase/Data_Base_v2"
    DB="/home/liiarpi-01/proyectopaltas/Local_data_base/Data_Base_v2"
    d_tt=transforms.Compose([
        phantom_segmentation(False),
        rgb_normalize(ImType=['PhantomRGB']),
        multi_image_resize(ImType=['PhantomRGB'],size=(100,100)),
        multi_ToTensor(ImType=['PhantomRGB']),
        output_transform()
        ])

    datab=Dataset_direct(root_dir=DB,ImType=['PhantomRGB'],Intersec=False,transform=d_tt)
    print("data loaded")
    device='cuda'

    #os.path.join("..","Data_prep")
    T_ID="GMVAE_A1_2"
    pth=os.path.join(str(pathlib.Path().absolute()),"results",T_ID)
    print(pth)

    model=GMVAE(image_dim=int(100),
        image_channels=3,
        repr_sizes=[6,12,24],
        layer_sizes=[10],
        w_latent_space_size=5,
        z_latent_space_size=5,
        y_latent_space_size=12,
        conv_kernel_size=7,
        conv_pooling=False,
        activators=[nn.Tanh(),nn.LeakyReLU(),nn.LeakyReLU()],
        conv_batch_norm=True,
        NN_batch_norm=True,
        stride=2,
        device="cpu")
    model.to(device)
    print("model loaded")

    tr=trainer(
        model=model,
        dataset=datab,
        epochs=30,
        folds=2,
        batch_size=10,
        use_cuda=True,
        loss_list=['conditional_prior','w_prior','y_prior','reconstruction',"total_loss"],
        data_dir=pth,
        in_device=None,
        num_workers=6,
    )

    tr.K_fold_train()
    
if __name__ == "__main__":
    fire.Fire(main)