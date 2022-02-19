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
    #DB="/run/user/1000/gvfs/afp-volume:host=MyCloudPR4100.local,user=aorus_1,volume=Paltas_DataBase/Data_Base_v2"
    #DB="//MYCLOUDPR4100/Paltas_DataBase/Data_Base_v2"
    DB="/home/liarpi2/proyectopaltas/Local_data_base/Data_Base_v2"
    d_tt=transforms.Compose([
        pos_fly_transform(ImType=["SenteraRGB"]),
        concatenate_non_uniform_transform(),
        ndvi_desc(),
        multi_image_resize(ImType=['SenteraNDVI'],size=(512,512)),
        multi_ToTensor(ImType=['SenteraNDVI']),
        only_tensor_transform()
        ])

    datab=Dataset_direct(root_dir=DB,ImType=['SenteraRGB','SenteraNIR','SenteraMASK'],Intersec=False,transform=d_tt)
    print("data loaded")
    device='cuda'

    #os.path.join("..","Data_prep")
    T_ID="GMVAE_A2_1"
    pth=os.path.join(str(pathlib.Path().absolute()),"results",T_ID)
    print(pth)

    model=GMVAE(image_dim=int(512),
        image_channels=1,
        non_uniform_dim=33,
        repr_sizes=[2,4,8,16,32,64],
        pre_layer_sizes=[300,200],
        layer_sizes=[100,50],
        pre_output=100,
        w_latent_space_size=8,
        z_latent_space_size=8,
        y_latent_space_size=12,
        conv_kernel_size=7,
        conv_pooling=False,
        activators=[nn.Sigmoid(),nn.LeakyReLU(),nn.LeakyReLU(),nn.LeakyReLU(),nn.LeakyReLU(),nn.LeakyReLU()],
        conv_batch_norm=True,
        NN_batch_norm=True,
        stride=2,
        device="cpu")
    model.to(device)
    print("model loaded")

    tr=trainer(
        model=model,
        dataset=datab,
        epochs=40,
        folds=2,
        batch_size=20,
        use_cuda=True,
        loss_list=['conditional_prior','w_prior','y_prior','reconstruction',"total_loss"],
        data_dir=pth,
        in_device=None,
        num_workers=3,
        args=["SenteraNDVI","Non_uniform_input"],
        uniform=False

    )

    tr.K_fold_train()
    
if __name__ == "__main__":
    fire.Fire(main)