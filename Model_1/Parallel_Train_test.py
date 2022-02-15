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
from Transforms import *

from torch import nn

def main():
    #DB="/run/user/1000/gvfs/afp-volume:host=MyCloudPR4100.local,user=paltas,volume=Paltas_DataBase/Data_Base_v2"
    DB="/home/lambda/paltas/Local_data_base/Data_Base_v2"
    #DB="//MYCLOUDPR4100/Paltas_DataBase/Data_Base_v2"
    d_tt=transforms.Compose([
        ndvi_desc(),
        multi_image_resize(ImType=['SenteraNDVI'],size=(512,512)),
        multi_ToTensor(ImType=['SenteraNDVI']),
        only_tensor_transform()
        ])

    datab=Dataset_direct(root_dir=DB,ImType=['SenteraRGB','SenteraNIR','SenteraMASK'],Intersec=False,transform=d_tt)
    print("data loaded")
    device='cuda:0'
    

    #os.path.join("..","Data_prep")
    T_ID="VAE_10"
    pth=os.path.join(str(pathlib.Path().absolute()),"results",T_ID)
    print(pth)

    model=b_encodeco(image_dim=int(512),
                 image_channels=1,
                 repr_sizes=[3,6,12,48,192],
                 layer_sizes=[200,100,50],
                 latent_space_size=50,
                 conv_kernel_size=15,
                 activators=[nn.Sigmoid(),nn.LeakyReLU(),nn.LeakyReLU(),nn.LeakyReLU(),nn.LeakyReLU()],
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
        batch_size=12,
        use_cuda=True,
        loss_list=['KLD','reconstruction',"total_loss"],
        data_dir=pth,
        in_device=device,
        num_workers=10,
        args=['SenteraNDVI']
    )

    tr.K_fold_train()
    
if __name__ == "__main__":
    fire.Fire(main)