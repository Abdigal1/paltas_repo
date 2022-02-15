import sys
import os

sys.path.append(os.path.join("..","Data_prep"))
sys.path.append(os.path.join("..","DL_utils"))

from Train_utils.TT_class import trainer

import pathlib
import fire as fire
from Net.parallel_VAE import b_encodeco
from torchvision import transforms
from Custom_dataloader import *
from Transforms import phantom_segmentation
from Transforms import multi_image_resize
from Transforms import multi_ToTensor
from Transforms import output_transform
from Transforms import rgb_normalize

from torch import nn

def main():
    #DB="/run/user/1000/gvfs/afp-volume:host=MyCloudPR4100.local,user=paltas,volume=Paltas_DataBase/Data_Base_v2"
    DB="/home/lambda/paltas/Local_data_base/Data_Base_v2"
    #DB="//MYCLOUDPR4100/Paltas_DataBase/Data_Base_v2"
    d_tt=transforms.Compose([
        phantom_segmentation(False,True),
        rgb_normalize(ImType=['PhantomRGB']),
        multi_image_resize(ImType=['PhantomRGB'],size=(512,512)),
        multi_ToTensor(ImType=['PhantomRGB']),
        output_transform()
        ])


#TO DEBUG
    #d_tt=transforms.Compose([
    #    phantom_segmentation(False,non_uniform_input=True),
    #    rgb_normalize(ImType=['PhantomRGB']),
    #    multi_image_resize(ImType=['PhantomRGB'],size=(20,20)),
    #    pos_fly_transform(),
    #    concatenate_non_uniform_transform(),
    #    multi_ToTensor(ImType=['PhantomRGB']),
    #    only_tensor_transform(),
    #    #output_transform()
    #    ])

    datab=Dataset_direct(root_dir=DB,ImType=['PhantomRGB'],Intersec=False,transform=d_tt)
    print("data loaded")
    device='cuda:0'
    

    #os.path.join("..","Data_prep")
    T_ID="VAE_6"
    pth=os.path.join(str(pathlib.Path().absolute()),"results",T_ID)
    print(pth)

    model=b_encodeco(image_dim=int(512),
                 image_channels=3,
                 non_uniform_input=30
                 repr_sizes=[12,48,192],
                 layer_sizes=[80,50],
                 latent_space_size=50,
                 conv_kernel_size=25,
                 activators=[nn.Sigmoid(),nn.ReLU(),nn.ReLU()],
                 conv_pooling=False,
                 conv_batch_norm=True,
                 NN_batch_norm=True,
                 stride=2,
                in_device=device)
    

    print("model loaded")

    tr=trainer(
        model=model,
        dataset=datab,
        epochs=30,
        folds=2,
        batch_size=10,
        use_cuda=True,
        loss_list=['KLD','reconstruction',"total_loss"],
        data_dir=pth,
        in_device=device,
        num_workers=10,
        args=["PhantomRGB","Non_uniform_input"],
        uniform=False
    )

    tr.K_fold_train()
    
if __name__ == "__main__":
    fire.Fire(main)