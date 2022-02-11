import sys
import os

sys.path.append(os.path.join("..","Data_prep"))
sys.path.append(os.path.join("..","DL_utils"))

from Train_utils.TT_class import trainer

import pathlib
import fire as fire
from Net.VAE_v2 import b_encodeco
from torchvision import transforms
from Custom_dataloader import *
from Transforms import phantom_segmentation
from Transforms import multi_image_resize
from Transforms import multi_ToTensor
from Transforms import output_transform
from Transforms import rgb_normalize

from torch import nn

def main():
#    DB="/run/user/1000/gvfs/afp-volume:host=MyCloudPR4100.local,user=aorus_1,volume=Paltas_DataBase/Data_Base_v2"
    DB="/home/liiarpi-01/proyectopaltas/Local_data_base/Data_Base_v2"
    #DB="//MYCLOUDPR4100/Paltas_DataBase/Data_Base_v2"

    device='cuda'

    d_tt=transforms.Compose([
        phantom_segmentation(False),
        #rgb_normalize(ImType=['PhantomRGB']),
        hue_transform(),
        multi_image_resize(ImType=['PhantomRGB'],size=(200,200)),
        multi_ToTensor(ImType=['PhantomRGB']),
        output_transform()
        ])

    model=b_encodeco(image_dim=int(200),
                 image_channels=1,
                 repr_sizes=[8,16,32],
                 layer_sizes=[100,50],
                 latent_space_size=25,
                 conv_kernel_size=20,
                 activators=[nn.Sigmoid(),nn.LeakyReLU(),nn.LeakyReLU()],
                 conv_pooling=False,
                 conv_batch_norm=True,
                 NN_batch_norm=True,
                 stride=2,
                device=device)
    model.to(device)

    datab=Dataset_direct(root_dir=DB,ImType=['PhantomRGB'],Intersec=False,transform=d_tt)
    print("data loaded")
    

    #os.path.join("..","Data_prep")
    T_ID="VAE_v2_6"
    pth=os.path.join(str(pathlib.Path().absolute()),"results",T_ID)
    print(pth)


    print("model loaded")

    #K_fold_train(model=model,
    #            dataset=datab,
    #            epochs=30,
    #            batch_size=2,
    #            use_cuda=True,
    #            folds=2,
    #            data_train_dir=pth,
    #            n_workers=6,
    #            loss_fn=MSEloss_fn_b
    # )

    tr=trainer(
        model=model,
        dataset=datab,
        epochs=30,
        folds=2,
        batch_size=4,
        use_cuda=True,
        loss_list=['KLD','reconstruction',"total_loss"],
        data_dir=pth,
        in_device=None,
        num_workers=6,
    )

    tr.K_fold_train()
    
if __name__ == "__main__":
    fire.Fire(main)