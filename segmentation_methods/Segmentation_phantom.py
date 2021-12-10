import numpy as np
import skimage
from skimage import io
from skimage.color import rgb2hsv
from skimage.morphology import binary_closing
import scipy

def seg_mask(img,squared=False):
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
    
    x_min=np.min(np.where(mask==1)[0])
    y_min=np.min(np.where(mask==1)[1])
    x_max=np.max(np.where(mask==1)[0])
    y_max=np.max(np.where(mask==1)[1])
    
    #if squared:
    smk=np.zeros(mask.shape)
    Lx=x_max-x_min
    Ly=y_max-y_min
    if Lx<smk.shape[1] and Ly<smk.shape[0]:
        if (Lx)>(Ly):
            d=np.floor((abs((Ly)-(Lx)))/2)
            dL=d
            dR=d+2*((abs((Ly)-(Lx)))/2-d)
            if (y_min-dL)<=0:
                y_min=0
                y_max=int(y_max+dR-(y_min-dL))
            elif (y_max+dR)>=smk.shape[0] or (y_max+dR)>=smk.shape[1]:
                y_max=smk.shape[1]
                y_min=int(y_min-dL+(y_max-dR))
            else:
                y_min=int(y_min-dL)
                y_max=int(y_max+dR)
        else:
            d=np.floor((abs((Ly)-(Lx))/2))
            dL=d
            dR=d+2*((abs((Ly)-(Lx)))/2-d)
            if (x_min-dL)<=0:
                x_min=0
                x_max=int(x_max+dR-(x_min-dL))
            elif (x_max+dR)>=smk.shape[0] or (x_max+dR)>=smk.shape[1]:
                x_max=smk.shape[0]
                x_min=int(x_min-dL+(x_max-dR))
            else:
                x_min=int(x_min-dL)
                x_max=int(x_max+dR)
    
    if squared:
        smk[x_min:x_max,y_min:y_max]=1
        mask=smk.astype("int")
        
    return mask,x_max,x_min,y_max,y_min

#DB="../../Labeled"
#Dates=os.listdir(DB)
#
#date_ID=15
#img_ID=16
#d=os.path.join(DB,Dates[date_ID])
#img_dir=os.path.join(d,os.listdir(d)[img_ID])
#img=io.imread(img_dir)
#
#for i in os.listdir(d):
#    img_dir=os.path.join(d,i)
#    img=io.imread(img_dir)
#    mk=seg_mask(img)
#    simg=np.stack((mk,mk,mk),axis=2)*img
#
#    fig, (ax1,ax2,ax3) = plt.subplots(1,3,figsize=(10,10))
#    ax1.imshow(img)
#    ax1.set_title('Original image')
#    ax2.imshow(mk)
#    ax2.set_title('Mask')
#    ax3.imshow(simg)
#    ax3.set_title('Segmented image')
#    plt.show()