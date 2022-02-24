import sys
import os

sys.path.append(os.path.join("..","Data_prep"))
sys.path.append(os.path.join("..","DL_utils"))

from Train_utils.TT_class import trainer

import pathlib
import fire as fire
from Net.VAE_v2 import GMVAE
from torchvision import transforms
from Custom_dataloader import *
from Transforms import phantom_segmentation
from Transforms import multi_image_resize
from Transforms import multi_ToTensor
from Transforms import output_transform
from Transforms import *

from torch import nn

def main():
    #DB="/run/user/1000/gvfs/afp-volume:host=MyCloudPR4100.local,user=paltas,volume=Paltas_DataBase/Data_Base_v2"
    DB="/home/lambda/paltas/Local_data_base/Data_Base_v2"
    
    d_tt=transforms.Compose([
        trans_registration(),
        multi_image_resize(ImType=['SenteraRGB','SenteraNIR'],size=(512,512)),
        stack_multiespectral(),
        multi_ToTensor(ImType=['SenteraRGBNIR']),
        only_tensor_transform()
        ])

    datab=Dataset_direct(root_dir=DB,ImType=['SenteraRGB','SenteraNIR','SenteraMASK'],Intersec=False,transform=d_tt)
    print("data loaded")
    device='cuda:0'
    

    #os.path.join("..","Data_prep")
    T_ID="GMVAE_L_1"
    pth=os.path.join(str(pathlib.Path().absolute()),"results",T_ID)
    print(pth)

    model=GMVAE(image_dim=int(512),
                 image_channels=5,
                 repr_sizes=[8,16,32,64,128],
                 layer_sizes=[200,100,50],
                w_latent_space_size=10,
                z_latent_space_size=10,
                y_latent_space_size=12,
                 conv_kernel_size=7,
                 activators=[nn.Sigmoid(),nn.LeakyReLU(),nn.LeakyReLU(),nn.LeakyReLU(),nn.LeakyReLU()],
                 conv_pooling=False,
                 conv_batch_norm=True,
                 NN_batch_norm=True,
                 stride=2,
                device="cuda",
                 Multi_GPU=True,
                 in_device="cuda:0"
                )
    

    print("model loaded")

    tr=trainer(
        model=model,
        dataset=datab,
        epochs=40,
        folds=2,
        batch_size=6,
        use_cuda=True,
        loss_list=['conditional_prior','w_prior','y_prior','reconstruction',"total_loss"],
        data_dir=pth,
        in_device=None,
        num_workers=10,
        args=["SenteraRGBNIR"]
    )

    tr.K_fold_train()
    
if __name__ == "__main__":
    fire.Fire(main)