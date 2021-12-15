import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
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
import matplotlib.pyplot as plt
from os import stat
import matplotlib.pyplot as plt
from skimage.feature import greycomatrix, greycoprops
from skimage.color import rgb2gray
from tqdm import tqdm
import pickle
def return_rec(bmask, small=True,fn = cv2.contourArea):
    bmask = bmask.astype(np.uint8)
    mass = np.zeros_like(bmask, dtype=np.uint8)
    _,contours ,_ =cv2.findContours(bmask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    rc = []
    for i in contours:
        epsilon = 0.5*cv2.arcLength(i,True)

        if cv2.contourArea(i)>8000:
            vv = cv2.boundingRect(i)
            vva = vv[2]*vv[3]
            if cv2.contourArea(i)/vva >=0.7:
                rc.append(i)
    rc = sorted(rc, key=fn)
    #print(len(rc))
    if small:
        cv2.fillPoly(mass, pts = [rc[0]], color=255)
    else:
        cv2.fillPoly(mass, pts = [rc[-1]], color=255)
    _,yy,_,hy = cv2.boundingRect(mass)
    
    return mass.astype(np.bool_), yy, hy

def get_clothe_masks(img):
    b_mask = ((img[:, :, 0]<20) & (img[:, :, 1]<20) & (img[:, :, 2]<20))
    b_mask, yb, hb = return_rec(b_mask)
    g_mask = (abs(img[:,:,2]-img[:,:,0])<14) & (img[:,:,2]>45)
    g_mask, yg, hg = return_rec(g_mask)
    y = min(yb, yg)
    h = max(hb, hg)
    ra = img.copy()
    ra[y+h+100::,:,2]=0
    ra[max(y-100,0)::-1,:,2]=0
    #plt.imshow(ra[:,:,::-1])
    r_mask = ((ra[:, :, 2]>105) & (ra[:, :, 1]<60))
    r_mask,_,_ = return_rec(r_mask, small=False)
    return r_mask, g_mask, b_mask

def reg(rgb_img, nir_img):
    ## Registration
    rgb_ = cv2.resize(rgb_img, (4000, 3000))
    rgb_ = cv2.cvtColor(rgb_, cv2.COLOR_BGR2GRAY) 
    
    nir_ = cv2.resize(nir_img, (4000, 3000))
    nir_ = nir_[:, :, 0]
    
    #akaze = cv2.xfeatures2d.SIFT_create()
    akaze = cv2.xfeatures2d.SURF_create() #python version mismatch
    kp1, des1 = akaze.detectAndCompute(rgb_, None)
    kp2, des2 = akaze.detectAndCompute(nir_, None)
    
    bf = cv2.BFMatcher(cv2.NORM_L1)
    matches = bf.knnMatch(des1, des2, k = 2)
    
    good_matches = []
    for m,n in matches:
        if m.distance < 0.8*n.distance:
            good_matches.append([m])
    
    ref_matched_kpts = np.float32([kp1[m[0].queryIdx].pt for m in good_matches])
    sensed_matched_kpts = np.float32([kp2[m[0].trainIdx].pt for m in good_matches])
    
    H, status = cv2.findHomography(sensed_matched_kpts, ref_matched_kpts, cv2.RANSAC,80.0)
    nir_img = cv2.warpPerspective(nir_img, H, (rgb_img.shape[1], rgb_img.shape[0]))
    return nir_img
def compute_ndvi(rgb_img, nir_img, mask, statistic=True):
    ## Registration
    rgb_ = cv2.resize(rgb_img, (4000, 3000))
    rgb_ = cv2.cvtColor(rgb_, cv2.COLOR_BGR2GRAY) 
    
    nir_ = cv2.resize(nir_img, (4000, 3000))
    nir_ = nir_[:, :, 0]
    
    akaze = cv2.xfeatures2d.SURF_create()
    
    kp1, des1 = akaze.detectAndCompute(rgb_, None)
    kp2, des2 = akaze.detectAndCompute(nir_, None)
    
    bf = cv2.BFMatcher(cv2.NORM_L1)
    matches = bf.knnMatch(des1, des2, k = 2)
    
    good_matches = []
    for m,n in matches:
        if m.distance < 0.8*n.distance:
            good_matches.append([m])
    
    ref_matched_kpts = np.float32([kp1[m[0].queryIdx].pt for m in good_matches])
    sensed_matched_kpts = np.float32([kp2[m[0].trainIdx].pt for m in good_matches])
    
    H, status = cv2.findHomography(sensed_matched_kpts, ref_matched_kpts, cv2.RANSAC,80.0)
    nir_img = cv2.warpPerspective(nir_img, H, (rgb_img.shape[1], rgb_img.shape[0]))
    
    ##Adjustment
    
    #====================SKIMAGE VERSION
    #rgb_img = rgb_img[::-1]
    #nir_img = nir_img[::-1]
    #=================================
    rgb_img = rgb_img.astype(np.float64)
    nir_img = nir_img.astype(np.float64)
    RR = np.array([[1.377, -0.182, -0.061],
          [-0.199, 1.420, -0.329],
          [-0.034, -0.110, 1.150]])
    RN = np.array([[-0.956, 0., 1.],
          [2.426, 0., -0.341],
          [0., 1., 0.]])
    maskn = ((nir_img[:,:,0]!=0)&(nir_img[:,:,1]!=0)&(nir_img[:,:,2]!=0)).astype(np.bool_)
    rgb_img = np.matmul(rgb_img, RR.T)
    nir_img = np.matmul(nir_img, RN.T)
    NDVI = (2.7*nir_img[:, :, 1]-rgb_img[:, :, 2])/(2.7*nir_img[:, :, 1]+rgb_img[:, :, 2])
    mask &= maskn
    ##Mean STD calculation
    #plt.imshow(NDVI)
    
    if not(statistic):
        #imout = np.zeros_like(NDVI)
        #imout[mask] = NDVI[mask]
        #return imout
        return cv2.bitwise_and(NDVI, NDVI, mask=mask)
    else:
        mask = mask.astype(bool)
        mndvi = NDVI[mask]
        return [mndvi.mean(), mndvi.std()]

class texture_desc():
    """
    Co-ocupance matrix analysis either ndvi(-1, 1) or rgb(0, 255)
    Working only in sentera(to be fixed)
    """
    def __init__(self,descriptor='correlation' ,from_rgb=True):
        self.descriptor = descriptor
        self.from_rgb = from_rgb
        
        
    def __call__(self, sample):
        img = np.zeros_like(sample['SenteraRGB'])
        if self.from_rgb:
            img = sample['SenteraRGB']
            
            img = rgb2gray(img)
            img = (img*255).astype(np.uint8)
        else:
            rgb, nir= sample['SenteraRGB'], sample['SenteraNIR']
            fmask = np.ones_like(rgb)[:,:,0]
            fmask = fmask.astype(np.bool_)
            rgb = rgb[::-1]
            nir = nir[::-1]
            ndvi = compute_ndvi(rgb,nir, fmask, statistic=False)
            img = ((ndvi+1)*255/2).astype(np.uint8)
        
        mask = sample['SenteraMASK']
        mask = mask.astype(bool)
        
        horizontal_indicies = np.where(np.any(mask, axis=0))[0]
        vertical_indicies = np.where(np.any(mask, axis=1))[0]
        x1, x2 = horizontal_indicies[[0, -1]]
        y1, y2 = vertical_indicies[[0, -1]]
        
        img = img[y1:y2, x1:x2]
        #img = img.astype(np.uint8)
            
        glcm = greycomatrix(img, distances=[15, 30, 50], angles=[0, np.pi/2], levels=256)
        
        return {('Sentera'+self.descriptor): greycoprops(glcm, self.descriptor), 'landmarks': sample['landmarks'], 'Date':sample['Date'], 'Place':sample['Place']}
 

DB="\\\MYCLOUDPR4100\\Paltas_DataBase\\Data_Base_v2"

d_t = transforms.Compose([texture_desc('correlation')])
datab=Dataset_direct(root_dir=DB,ImType=['SenteraRGB', 'SenteraMASK'],Intersec=False, transform=d_t)
print('Data cargada')


for i, item in tqdm(enumerate(datab)):
    with open('senterargbcorrelation_v2.pkl', 'ab') as f:
        pickle.dump(item, f)



DB="\\\MYCLOUDPR4100\\Paltas_DataBase\\Data_Base_v2"

d_t = transforms.Compose([texture_desc('energy')])
datab=Dataset_direct(root_dir=DB,ImType=['SenteraRGB', 'SenteraMASK'],Intersec=False, transform=d_t)
print('Data cargada')

for i, item in tqdm(enumerate(datab)):
    with open('senterargbenergy_v2.pkl', 'ab') as f:
        pickle.dump(item, f)

DB="\\\MYCLOUDPR4100\\Paltas_DataBase\\Data_Base_v2"

d_t = transforms.Compose([texture_desc('ASM')])
datab=Dataset_direct(root_dir=DB,ImType=['SenteraRGB', 'SenteraMASK'],Intersec=False, transform=d_t)
print('Data cargada')

for i, item in tqdm(enumerate(datab)):
    with open('senterargbasm_v2.pkl', 'ab') as f:
        pickle.dump(item, f)

DB="\\\MYCLOUDPR4100\\Paltas_DataBase\\Data_Base_v2"

d_t = transforms.Compose([texture_desc('homogeneity')])
datab=Dataset_direct(root_dir=DB,ImType=['SenteraRGB', 'SenteraMASK'],Intersec=False, transform=d_t)
print('Data cargada')

for i, item in tqdm(enumerate(datab)):
    with open('senterargbhomogeneity_v2.pkl', 'ab') as f:
        pickle.dump(item, f)

DB="\\\MYCLOUDPR4100\\Paltas_DataBase\\Data_Base_v2"

d_t = transforms.Compose([texture_desc('dissimilarity')])
datab=Dataset_direct(root_dir=DB,ImType=['SenteraRGB', 'SenteraMASK'],Intersec=False, transform=d_t)
print('Data cargada')

print('Disimilarity')
for i, item in tqdm(enumerate(datab)):
    with open('senterargbdissimilarity_v2.pkl', 'ab') as f:
        pickle.dump(item, f)


DB="\\\MYCLOUDPR4100\\Paltas_DataBase\\Data_Base_v2"

d_t = transforms.Compose([texture_desc('contrast')])
datab=Dataset_direct(root_dir=DB,ImType=['SenteraRGB', 'SenteraMASK'],Intersec=False, transform=d_t)
print('Data cargada')

for i, item in tqdm(enumerate(datab)):
    with open('senterargbcontrast_v2.pkl', 'ab') as f:
        pickle.dump(item, f)