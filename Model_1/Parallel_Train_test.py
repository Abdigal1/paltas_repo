import pathlib
import fire as fire
from B_VAE.parallel_VAE import b_encodeco
from Train_utils import train_utils
from B_VAE.Utils_imp_VAE import loss_fn_b
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

def main():
    #DB="/run/user/1000/gvfs/afp-volume:host=MyCloudPR4100.local,user=paltas,volume=Paltas_DataBase/Data_Base_v2"
    DB="/home/lambda/paltas/Local_data_base/Data_Base_v2"
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
    device='cuda:0'
    

    #os.path.join("..","Data_prep")
    T_ID="VAE_4"
    pth=os.path.join(str(pathlib.Path().absolute()),"results",T_ID)
    print(pth)

    model=b_encodeco(image_dim=int(200),
                 image_channels=3,
                 repr_sizes=[12,48,192],
                 layer_sizes=[100],
                 latent_space_size=30,
                 conv_kernel_size=35,
                 conv_pooling=False,
                 conv_batch_norm=True,
                 NN_batch_norm=True,
                 stride=2,
                in_device=device)
    

    print("model loaded")

    try:
        K_fold_train(model=model,
                dataset=datab,
                epochs=30,
                batch_size=10,
                use_cuda=True,
                folds=2,
                data_train_dir=pth,
                loss_fn=loss_fn_b,
                in_device=device
        )
    except IndexError as e:
        print(e)
    return 0
    
if __name__ == "__main__":
    fire.Fire(main)