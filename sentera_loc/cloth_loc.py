import cv2
import numpy as np
import os
import shutil

PATH = '/home/lambda/paltas/repo/Sentera_Data_Base'
DPATH = '/home/lambda/paltas/repo/telas'
n = 0
for i in os.listdir(PATH):
    l1 = os.path.join(PATH, i, 'RGB')
    m1 = os.path.join(PATH, i, 'NIR')
    for j in os.listdir(l1):
        l2 = os.path.join(l1, j)
        m2 = os.path.join(m1, j)
        if j.endswith('.jpg'):
            a = cv2.imread(l2)
            if a is None:
                continue
            b = (a[:, :, 2] > 150) & (a[:, :, 1] < 60)
            if np.sum(np.sum(b))/(a.shape[0]*a.shape[1]) > 0.007:
                n += 1
                shutil.copy(l2, os.path.join(DPATH, 'tela_RGB'+str(n)+'.jpg'))
                shutil.copy(m2, os.path.join(DPATH, 'tela_NIR'+str(n)+'.jpg'))
