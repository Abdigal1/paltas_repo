import pathlib
import fire as fire
from B_VAE.B_VAE import b_encodeco
from Train_utils import train_utils
from B_VAE.Utils_VAE import loss_fn
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

import torch

def main():
    DB="/run/user/1000/gvfs/afp-volume:host=MyCloudPR4100.local,user=paltas,volume=Paltas_DataBase/Data_Base_v2"
    #DB="//MYCLOUDPR4100/Paltas_DataBase/Data_Base_v2"
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
    T_ID="VAE_1"
    pth=os.path.join(str(pathlib.Path().absolute()),"results",T_ID)
    print(pth)

    model=b_encodeco(image_dim=int(100),
                 image_channels=3,
                 repr_sizes=[15,30],
                 layer_sizes=[50],
                 latent_space_size=5,
                device=device)
    model.to(device)
    print("model loaded")

    K_fold_train(model=model,
                dataset=datab,
                epochs=20,
                batch_size=10,
                use_cuda=True,
                folds=2,
                data_train_dir=pth,
                loss_fn=loss_fn
     )
    
if __name__ == "__main__":
    fire.Fire(main)