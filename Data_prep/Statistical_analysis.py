import os
#import skimage
import torch
from torchvision import transforms
import numpy as np
import glob
from skimage import io
import skimage
import matplotlib.pyplot as plt
from Custom_dataloader import *
from Transforms import phantom_segmentation
from Transforms import entropy_mark_transform
from Transforms import hsv_stats_transfrom
from Transforms import lab_stats_transfrom
from Transforms import black_perc_transfrom
import matplotlib.pyplot as plt
import pickle

DB="/run/user/1000/gvfs/afp-volume:host=MyCloudPR4100.local,user=admin,volume=Paltas_DataBase/Data_Base"
savein="/run/user/1000/gvfs/afp-volume:host=MyCloudPR4100.local,user=admin,volume=Paltas_DataBase/metadata"
d_t=transforms.Compose([phantom_segmentation(False)])
datab=Dataset_direct(root_dir=DB,ImType=['PhantomRGB'],Intersec=False,transform=d_t)
t1=black_perc_transfrom()
t2=hsv_stats_transfrom()
t3=lab_stats_transfrom()

T1={}
T2={}
T3={}

for i in range(len(datab.aID)):
#for i in range(2):
    print(datab.aID[i])
    T1[datab.aID[i]]=t1(datab[i])['bl_per']
    T2[datab.aID[i]]=t2(datab[i])['stat_val']
    T3[datab.aID[i]]=t3(datab[i])['stat_val']

f1=open("/run/user/1000/gvfs/afp-volume:host=MyCloudPR4100.local,user=admin,volume=Paltas_DataBase/metadata/bl_per_wmask_phantom.pkl",'wb')
f2=open("/run/user/1000/gvfs/afp-volume:host=MyCloudPR4100.local,user=admin,volume=Paltas_DataBase/metadata/stat_val_hsv_wmask_phantom.pkl",'wb')
f3=open("/run/user/1000/gvfs/afp-volume:host=MyCloudPR4100.local,user=admin,volume=Paltas_DataBase/metadata/stat_val_lab_wmask_phantom.pkl",'wb')

pickle.dump(T1,f1)
pickle.dump(T2,f2)
pickle.dump(T3,f3)

f1.close()
f2.close()
f3.close()