import numpy as np
from skimage import io
import skimage


def hist_stat_h(img):
    img_hsv=skimage.color.rgb2hsv(img)
    imgr=img_hsv[:,:,0].ravel()[img_hsv[:,:,0].ravel()>0.00001]
    m=np.mean(imgr)
    st=np.std(imgr)
    v,b=np.histogram(imgr.ravel(),bins=100)
    mo=b[np.argmax(v)]
    return m,st,mo

def hist_stat_l(img):
    img_hsv=skimage.color.rgb2lab(img.astype('uint8'))
    imgr=img_hsv[:,:,0].ravel()[np.logical_and(img_hsv[:,:,0].ravel()>0.00001,img_hsv[:,:,0].ravel()<10)]
    m=np.mean(imgr)
    st=np.std(imgr)
    v,b=np.histogram(imgr.ravel(),bins=100)
    mo=b[np.argmax(v)]
    return m,st,mo

def bl_perc(img):
    img_hsv=skimage.color.rgb2lab(img.astype('uint8'))
    imgr=img_hsv[:,:,0].ravel()[np.logical_and(img_hsv[:,:,0].ravel()>0.00001,img_hsv[:,:,0].ravel()<10)]
    m=(imgr.shape[0])/((img_hsv[:,:,0].ravel()[img_hsv[:,:,0].ravel()>0.00001]).shape[0])
    return m

#entropy
def entropy_mark(img):
    img=skimage.color.rgb2lab(img.astype('uint8'))[:,:,0]
    ei=(skimage.filters.rank.entropy(img/128,skimage.morphology.disk(5)))>4
    return ei