import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
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

def compute_ndvi(rgb_img, nir_img, mask, statistic=True):
    ## Registration
    #rgb_ = cv2.resize(rgb_img, (600, 450))
    rgb_ = rgb_img
    #if rgb_.dtype != np.uint8:
    #    rgb_ = rgb_.astype(np.uint8)
    rgb_ = cv2.cvtColor(rgb_, cv2.COLOR_BGR2GRAY) 
    nir_ = nir_img
    #nir_ = cv2.resize(nir_img, (600, 450))
    #if nir_.dtype != np.uint8:
    #    nir_ = nir_.astype(np.uint8)
    nir_ = nir_[:, :, 0]
    
    akaze = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = akaze.detectAndCompute(rgb_, None)
    kp2, des2 = akaze.detectAndCompute(nir_, None)
    
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k = 2)
    
    good_matches = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good_matches.append([m])
    #print(ref_matched_kpts, sensed_matched_kpts)
    ref_matched_kpts = np.float32([kp1[m[0].queryIdx].pt for m in good_matches])
    sensed_matched_kpts = np.float32([kp2[m[0].trainIdx].pt for m in good_matches])
    
    H, status = cv2.findHomography(sensed_matched_kpts, ref_matched_kpts, cv2.RANSAC,80.0)
    nir_img = cv2.warpPerspective(nir_img, H, (rgb_img.shape[1], rgb_img.shape[0]))
    
    ##Adjustment
    rgb_img = rgb_img.astype(np.float64)
    nir_img = nir_img.astype(np.float64)
    RR = np.array([[1.377, -0.182, -0.061],
          [-0.199, 1.420, -0.29],
          [-0.034, -0.110, 1.150]])
    RN = np.array([[-0.956, 0., 1.],
          [2.426, 0., -0.341],
          [0., 1., 0.]])
    
    rgb_img = np.matmul(rgb_img, RR)
    nir_img = np.matmul(nir_img, RN)
    NDVI = (2.7*nir_img[:, :, 1]-rgb_img[:, :, 2])/(2.7*nir_img[:, :, 1]+rgb_img[:, :, 2])
    mask = mask.astype(bool)
    ##Mean STD calculation
    #plt.imshow(NDVI)
    
    if not(statistic):
        mask = mask.astype(np.uint8)
        return cv2.bitwise_and(NDVI, NDVI, mask=mask)
    else:
        mndvi = NDVI[mask]
        return [mndvi.mean(), mndvi.std()]

#if __name__ == '__main__':
#    a = cv2.imread('..\\Tests\\tela_RGB7.jpg')
#    b = cv2.imread('..\\Tests\\tela_NIR7.jpg')
#    print([compute_ndvi(a,b, i) for i in get_clothe_masks(a)])