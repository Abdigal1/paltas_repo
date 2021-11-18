import numpy as np
import skimage
from skimage import io
from skimage.color import rgb2hsv
from skimage.morphology import binary_closing
import os
import matplotlib.pyplot as plt
import scipy

def seg_mask(img):
    hsv_i=rgb2hsv(img)
    h=hsv_i[:,:,0]
    h_b=np.logical_and(h>0.2,h<0.4)
    
    #Erosion and dilation
    disk=skimage.morphology.disk(5)
    hl=np.ones((1,100)).astype('uint8')
    cimg=skimage.morphology.erosion(h_b,disk)
    cimg=binary_closing(cimg,hl)
    cimg=binary_closing(cimg,hl.T)

    limg=skimage.measure.label(cimg)
    props=skimage.measure.regionprops(limg)

    cimg=((limg==np.argmax(np.vectorize(lambda p:p.area)(np.array(props)))+1)).astype("int")

    #Fill
    mask=scipy.ndimage.binary_fill_holes(cimg).astype("int")
    return mask

DB="../../Labeled"
Dates=os.listdir(DB)

date_ID=15
img_ID=16
d=os.path.join(DB,Dates[date_ID])
img_dir=os.path.join(d,os.listdir(d)[img_ID])
img=io.imread(img_dir)

for i in os.listdir(d):
    img_dir=os.path.join(d,i)
    img=io.imread(img_dir)
    mk=seg_mask(img)
    simg=np.stack((mk,mk,mk),axis=2)*img

    fig, (ax1,ax2,ax3) = plt.subplots(1,3,figsize=(10,10))
    ax1.imshow(img)
    ax1.set_title('Original image')
    ax2.imshow(mk)
    ax2.set_title('Mask')
    ax3.imshow(simg)
    ax3.set_title('Segmented image')
    plt.show()