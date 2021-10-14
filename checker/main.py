import os
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
import matplotlib.pyplot as plt
import numpy as np
import shutil

def getim_meta(X):
    info = X._getexif()
    GPSDATA = info[list(TAGS.keys())[list(TAGS.values()).index('GPSInfo')]]
    #Latitud, Longitud, Altitud
    return (GPSDATA[2][2].__float__(), GPSDATA[4][2].__float__(), GPSDATA[6].__float__())
class img_set():
    def __init__(self, path = '..\\..\\unlabeled_images'):
        self.path = path
        self.imgs_path = os.listdir(self.path)

    def __len__(self):
        return len(self.imgs_path)

    def __getitem__(self, index):
        img = self.imgs_path[index]
        full_path = os.path.join(self.path, img)
        X = Image.open(full_path)
        y = img
        return X, y, full_path
    
    def get_3dcoordinates(self):
        val = map(lambda i: getim_meta(self.__getitem__(i)[0]), list(range(self.__len__())))
        val = np.array(list(val))
        return val
    
    def get_2dcoordinates(self):
        val = self.get_3dcoordinates()
        plane = val[: ,:2]
        return plane
    
    def to_plot(self, MAX_X = 400):
        plane = self.get_2dcoordinates()

        return plane[:, 0], plane[:, 1]
    
    def plot(self, res_ind = None):
        xp, yp = self.to_plot()
        if res_ind is not None:
            xx, yy, _ = getim_meta(self.__getitem__(res_ind)[0])
            print(self.imgs_path[res_ind])
            fig = plt.figure()
            ax1 = fig.add_subplot(1,2,1)
            ax1.scatter(xp, yp)
            ax1.scatter(xx, yy)
            ax2 = fig.add_subplot(1,2,2)
            ax2.imshow(self.__getitem__(res_ind)[0])
            plt.show()

        else:
            plt.scatter(xp, yp)
            plt.show()
    
    def revision(self):
        os.mkdir('..\\Revisado')
        for i in range(self.__len__()):

            op = self.__getitem__(i)[2]
            self.plot(i)
            d = input()
            if d == 's':
                continue
            # n_m n es arbol, m es fila 
            # 1_5 arbol 1 fila 5
            n_p = os.path.join('..\\Revisado', 'arbol_' + d.split('_')[0] + 'fila_' + d.split('_')[1]+'.JPG')
            print(n_p)
            shutil.copy(op, n_p)

            

if __name__ == '__main__':
    lab = img_set(path = '..\\labeled_images')
    lab.revision()
        

