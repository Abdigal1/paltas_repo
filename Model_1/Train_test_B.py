import pathlib
import fire as fire
from B_VAE.VAE_v2 import b_encodeco
from Train_utils import train_utils
from B_VAE.Utils_imp_VAE import MSEloss_fn_b
from Train_utils.train_utils import train,test,K_fold_train

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
from torch import nn

def main():
#    DB="/run/user/1000/gvfs/afp-volume:host=MyCloudPR4100.local,user=aorus_1,volume=Paltas_DataBase/Data_Base_v2"
    DB="/home/liiarpi-01/proyectopaltas/Local_data_base/Data_Base_v2"
    #DB="//MYCLOUDPR4100/Paltas_DataBase/Data_Base_v2"
    d_tt=transforms.Compose([
        phantom_segmentation(False),
        rgb_normalize(ImType=['PhantomRGB']),
        multi_image_resize(ImType=['PhantomRGB'],size=(200,200)),
        multi_ToTensor(ImType=['PhantomRGB']),
        output_transform()
        ])

    datab=Dataset_direct(root_dir=DB,ImType=['PhantomRGB'],Intersec=False,transform=d_tt)
    print("data loaded")
    device='cuda'

    #os.path.join("..","Data_prep")
    T_ID="VAE_v2_5"
    pth=os.path.join(str(pathlib.Path().absolute()),"results",T_ID)
    print(pth)

    model=b_encodeco(image_dim=int(200),
                 image_channels=3,
                 repr_sizes=[10,20,40],
                 layer_sizes=[],
                 latent_space_size=50,
                 conv_kernel_size=15,
                 activators=[nn.Tanh(),nn.ReLU(),nn.ReLU()],
                 conv_pooling=False,
                 conv_batch_norm=True,
                 NN_batch_norm=True,
                 stride=2,
                device=device)
    model.to(device)
    print("model loaded")

    K_fold_train(model=model,
                dataset=datab,
                epochs=30,
                batch_size=2,
                use_cuda=True,
                folds=2,
                data_train_dir=pth,
                n_workers=6,
                loss_fn=MSEloss_fn_b
     )
    
if __name__ == "__main__":
    fire.Fire(main)