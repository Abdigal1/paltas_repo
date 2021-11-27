import cv2
import os
import numpy as np

def registration(rgb_img, nir_img):
    im1 = cv2.imread(rgb_img)
    im2 = cv2.imread(nir_img)
    im1p = cv2.resize(im1, (600, 450))
    im2p = cv2.resize(im2, (600, 450))
    im1p = cv2.cvtColor(im1p, cv2.COLOR_BGR2GRAY)
    im2p = im2p[:, :,2 ]
    akaze = cv2.xfeatures2d.SIFT_create()

    kp1, des1 = akaze.detectAndCompute(im1p, None)
    kp2, des2 = akaze.detectAndCompute(im2p, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k = 2)


    for m,n in matches:

        if m.distance < 0.75*n.distance:
            good_matches.append([m])
    
    ref_matched_kpts = np.float32([kp1[m[0].queryIdx].pt for m in good_matches])
    sensed_matched_kpts = np.float32([kp2[m[0].trainIdx].pt for m in good_matches])
    #print(len(ref_matched_kpts), len(sensed_matched_kpts))
    # Compute homography
    H, status = cv2.findHomography(sensed_matched_kpts, ref_matched_kpts, cv2.RANSAC,80.0)

    warped_image = cv2.warpPerspective(im2, H, (im2.shape[1], im2.shape[0]))
    
    return warped_image

def find_clothe(rgb_img):
    mask = () & ()
    
    
    return mask

def extract_ndvi(rgb_img, nir_img, mask):




if __name__ == '__main__':
    registration()