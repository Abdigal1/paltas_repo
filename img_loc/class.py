import os
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
import matplotlib.pyplot as plt
import numpy as np


class img_set():
    def __init__(self, path = '..\\unlabeled_images'):
        self.path = path
        self.imgs_path = os.listdir(self.path)

    def __len__(self):
        return len(self.imgs_path)

    def __getitem__(self, index):
        img = self.imgs_path[index]
        X = Image.open(os.path.join(self.path, img))
        y = img
        return X, y

    @staticmethod
    def getim_meta(X):
        info = X._getexif()
        GPSDATA = info[list(TAGS.keys())[list(TAGS.values()).index('GPSInfo')]]
        #Latitud, Longitud, Altitud
        return (GPSDATA[2][2], GPSDATA[4][2], GPSDATA[6])

    def plot(self):
        val = map(lambda i: getim_meta(self.__getitem__(i)[0]), list(range(self.__len__())))
        val = np.array(list(val))
        MAX_X = 400.0
        plane = val[: ,:2]
        #print(plane)
        minx = min(plane[:, 0])
        maxx = max(plane[:, 0])
        minx = minx.__float__()
        maxx = maxx.__float__()

        #print(maxx, minx, dir(minx))
        scale = MAX_X/(maxx-minx)
        #print(scale)
        plane *= scale
        plane -= minx*scale

        plt.scatter(plane[:, 0], plane[:, 1])
        plt.show()

