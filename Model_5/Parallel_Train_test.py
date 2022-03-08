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
from Transforms import output_transform
from Transforms import *

from torch import nn

def main():
    #DB="/run/user/1000/gvfs/afp-volume:host=MyCloudPR4100.local,user=paltas,volume=Paltas_DataBase/Data_Base_v2"
    DB="/home/lambda/paltas/Local_data_base/Data_Base_v2"
    #DB="//MYCLOUDPR4100/Paltas_DataBase/Data_Base_v2"
    d_tt=transforms.Compose([
        trans_registration(),
        multi_image_resize(ImType=['SenteraRGB','SenteraNIR'],size=(512,512)),
        stack_multiespectral(),
        pos_fly_transform(ImType=["SenteraRGB"]),
        concatenate_non_uniform_transform(),
        multi_ToTensor(ImType=['SenteraRGBNIR']),
        only_tensor_transform()
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

    datab=Dataset_direct(root_dir=DB,ImType=['SenteraRGB','SenteraNIR','SenteraMASK'],Intersec=False,transform=d_tt)
    print("data loaded")
    device='cuda:0'
    

    #os.path.join("..","Data_prep")
    T_ID="metaVAE_L_2"
    pth=os.path.join(str(pathlib.Path().absolute()),"results",T_ID)
    print(pth)

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
        args=["SenteraRGBNIR","Non_uniform_input"],
        uniform=False
    )

    tr.K_fold_train()
    
if __name__ == "__main__":
    fire.Fire(main)