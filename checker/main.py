import os
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
import matplotlib.pyplot as plt
import numpy as np
import shutil 
from shutil import copy2

def getim_meta(X):
    info = X._getexif()
    GPSDATA = info[list(TAGS.keys())[list(TAGS.values()).index('GPSInfo')]]
    #Latitud, Longitud, Altitud
    return (GPSDATA[2][2].__float__(), GPSDATA[4][2].__float__(), GPSDATA[6].__float__())
class img_set():
    def __init__(self, path = '..\\..\\unlabeled_images'):
        self.path = path
        self.imgs_path = os.listdir(self.path)
        self.plane = None
        self.rd = {'arbol_13_fila_F': (29.5011, 48.7047),
                    'arbol_4_fila_F': (29.0322, 46.733),
                    'arbol_8_fila_D': (28.7721, 47.5987),
                    'arbol_2_fila_F': (28.9472, 46.3624),
                    'arbol_19_fila_E': (29.554, 49.9057),
                    'arbol_4_fila_A': (27.9154, 46.973),
                    'arbol_10_fila_A': (28.2613, 48.3895),
                    'arbol_11_fila_G': (29.6306, 48.267),
                    'arbol_17_fila_E': (29.4646, 49.5296),
                    'arbol_10_fila_D': (28.9165, 48.2267),
                    'arbol_1_fila_C': (28.2432, 46.2765),
                    'arbol_3_fila_B': (28.0905, 46.7139),
                    'arbol_2_fila_D': (28.4968, 46.4519),
                    'arbol_4_fila_E': (28.8185, 46.8451),
                    'arbol_13_fila_A': (28.4009, 48.9501),
                    'arbol_4_fila_D': (28.5933, 46.8515),
                    'arbol_19_fila_B': (28.8791, 49.9889),
                    'arbol_11_fila_E': (29.1864, 48.3719),
                    'arbol_3_fila_A': (27.8659, 46.7661),
                    'arbol_15_fila_B': (28.7033, 49.2775),
                    'arbol_20_fila_E': (29.5935, 50.0725),
                    'arbol_7_fila_E': (28.9468, 47.384),
                    'arbol_13_fila_G': (29.7247, 48.6617),
                    'arbol_8_fila_F': (29.219, 47.5082),
                    'arbol_1_fila_B': (28.0022, 46.3406),
                    'arbol_5_fila_G': (29.3058, 46.907),
                    'arbol_1_fila_E': (28.661, 46.1751),
                    'arbol_13_fila_B': (28.6067, 48.8612),
                    'arbol_17_fila_B': (28.7871, 49.6215),
                    'arbol_11_fila_D': (28.9556, 48.3928),
                    'arbol_18_fila_C': (29.0752, 49.8126),
                    'arbol_17_fila_G': (29.908, 49.4172),
                    'arbol_5_fila_C': (28.4275, 47.0776),
                    'arbol_4_fila_B': (28.1369, 46.9011),
                    'arbol_3_fila_E': (28.7662, 46.6325),
                    'arbol_12_fila_E': (29.2331, 48.5658),
                    'arbol_8_fila_C': (28.5622, 47.6471),
                    'arbol_20_fila_B': (28.9223, 50.1752),
                    'arbol_9_fila_G': (29.4911, 47.6732),
                    'arbol_16_fila_D': (29.1858, 49.3688),
                    'arbol_9_fila_F': (29.2674, 47.7152),
                    'arbol_16_fila_F': (29.6364, 49.2844),
                    'arbol_20_fila_F': (29.8159, 50.0261),
                    'arbol_8_fila_G': (29.4443, 47.4781),
                    'arbol_7_fila_G': (29.3888, 47.2475),
                    'arbol_15_fila_A': (28.4934, 49.3512),
                    'arbol_6_fila_G': (29.3534, 47.1024),
                    'arbol_8_fila_E': (28.9962, 47.5822),
                    'arbol_21_fila_E': (29.6359, 50.2426),
                    'arbol_16_fila_C': (28.9782, 49.4125),
                    'arbol_1_fila_A': (27.769, 46.3702),
                    'arbol_12_fila_B': (28.5639, 48.6853),
                    'arbol_10_fila_C': (28.7121, 48.2885),
                    'arbol_20_fila_C': (29.1601, 50.1803),
                    'arbol_18_fila_D': (29.2726, 49.7403),
                    'arbol_5_fila_E': (28.8534, 46.991),
                    'arbol_6_fila_A': (28.0164, 47.3916),
                    'arbol_17_fila_D': (29.2253, 49.5393),
                    'arbol_5_fila_D': (28.6356, 47.0233),
                    'arbol_12_fila_C': (28.798, 48.6532),
                    'arbol_14_fila_A': (28.4428, 49.131),
                    'arbol_17_fila_F': (29.6794, 49.4516),
                    'arbol_10_fila_F': (29.355, 48.1108),
                    'arbol_2_fila_E': (28.7129, 46.4131),
                    'arbol_14_fila_B': (28.6498, 49.0515),
                    'arbol_19_fila_D': (29.3196, 49.936),
                    'arbol_17_fila_A': (28.5834, 49.7174),
                    'arbol_21_fila_D': (29.4124, 50.2791),
                    'arbol_16_fila_G': (29.8615, 49.2324),
                    'arbol_7_fila_D': (28.7245, 47.4123),
                    'arbol_9_fila_C': (28.6057, 47.8398),
                    'arbol_6_fila_C': (28.4718, 47.2609),
                    'arbol_21_fila_C': (29.2002, 50.3258),
                    'arbol_8_fila_A': (28.098, 47.7177),
                    'arbol_12_fila_F': (29.447, 48.4867),
                    'arbol_15_fila_C': (28.9331, 49.2263),
                    'arbol_14_fila_E': (29.3201, 48.9355),
                    'arbol_9_fila_E': (29.0434, 47.7803),
                    'arbol_14_fila_D': (29.0934, 48.9761),
                    'arbol_4_fila_G': (29.2522, 46.6857),
                    'arbol_9_fila_B': (28.3716, 47.8764),
                    'arbol_2_fila_A': (27.8247, 46.5914),
                    'arbol_19_fila_A': (28.6777, 50.1117),
                    'arbol_2_fila_G': (29.1619, 46.3259),
                    'arbol_8_fila_B': (28.3286, 47.6995),
                    'arbol_18_fila_G': (29.9548, 49.6141),
                    'arbol_21_fila_A': (28.764, 50.4527),
                    'arbol_4_fila_C': (28.3831, 46.887),
                    'arbol_5_fila_B': (28.1846, 47.104),
                    'arbol_11_fila_A': (28.3045, 48.5741),
                    'arbol_15_fila_D': (29.1384, 49.1635),
                    'arbol_17_fila_C': (29.0247, 49.6146),
                    'arbol_14_fila_G': (29.7665, 48.8343),
                    'arbol_21_fila_G': (30.0777, 50.1261),
                    'arbol_18_fila_A': (28.6305, 49.9153),
                    'arbol_1_fila_D': (28.4524, 46.2644),
                    'arbol_18_fila_B': (28.8389, 49.8347),
                    'arbol_12_fila_G': (29.6708, 48.4426),
                    'arbol_16_fila_E': (29.4122, 49.3191),
                    'arbol_11_fila_B': (28.5216, 48.5127),
                    'arbol_21_fila_F': (29.8621, 50.1728),
                    'arbol_10_fila_B': (28.4781, 48.3359),
                    'arbol_6_fila_E': (28.9018, 47.195),
                    'arbol_12_fila_D': (29.0062, 48.6035),
                    'arbol_9_fila_A': (28.1471, 47.9153),
                    'arbol_16_fila_B': (28.7457, 49.4452),
                    'arbol_6_fila_B': (28.2302, 47.2912),
                    'arbol_20_fila_D': (29.357, 50.1062),
                    'arbol_21_fila_B': (28.9737, 50.3557),
                    'arbol_14_fila_F': (29.542, 48.8773),
                    'arbol_6_fila_D': (28.6803, 47.2253),
                    'arbol_6_fila_F': (29.1248, 47.1186),
                    'arbol_19_fila_G': (29.9995, 49.8092),
                    'arbol_15_fila_E': (29.3661, 49.1288),
                    'arbol_20_fila_G': (30.0451, 50.0072),
                    'arbol_15_fila_F': (29.59, 49.0824),
                    'arbol_13_fila_D': (29.0541, 48.8035),
                    'arbol_18_fila_F': (29.7242, 49.6459),
                    'arbol_19_fila_C': (29.1124, 49.9873),
                    'arbol_7_fila_F': (29.1714, 47.3223),
                    'arbol_15_fila_G': (29.8138, 49.031),
                    'arbol_18_fila_E': (29.5046, 49.6971),
                    'arbol_2_fila_B': (28.0431, 46.5199),
                    'arbol_5_fila_A': (27.9647, 47.1841),
                    'arbol_13_fila_E': (29.2747, 48.7379),
                    'arbol_1_fila_G': (29.1196, 46.1038),
                    'arbol_11_fila_C': (28.7606, 48.4832),
                    'arbol_10_fila_E': (29.1431, 48.1946),
                    'arbol_3_fila_F': (28.9818, 46.5415),
                    'arbol_7_fila_C': (28.5186, 47.465),
                    'arbol_5_fila_F': (29.0725, 46.9139),
                    'arbol_13_fila_C': (28.8384, 48.8303),
                    'arbol_1_fila_F': (28.8958, 46.1537),
                    'arbol_3_fila_D': (28.5416, 46.6454),
                    'arbol_20_fila_A': (28.7176, 50.2685),
                    'arbol_9_fila_D': (28.82, 47.8044),
                    'arbol_7_fila_B': (28.2829, 47.5049),
                    'arbol_14_fila_C': (28.8849, 49.0122),
                    'arbol_16_fila_A': (28.5382, 49.5311),
                    'arbol_2_fila_C': (28.2924, 46.5181),
                    'arbol_3_fila_G': (29.2136, 46.5316),
                    'arbol_7_fila_A': (28.0496, 47.5304),
                    'arbol_11_fila_F': (29.4034, 48.3055),
                    'arbol_3_fila_C': (28.3362, 46.6955),
                    'arbol_19_fila_F': (29.7729, 49.8489)}

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
        self.plane = val[: ,:2]
        return self.plane
    
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
        myfile = open('skipped.txt', 'w')
        try :
            os.mkdir('../Revisado')
        except:
            pass
        for i in range(self.__len__()):

            op = self.__getitem__(i)[2]
            self.plot(i)
            d = input()
            if d == 's':
                myfile.write("%s\n" % op)
                continue
            # n_m n es arbol, m es fila 
            # 1_A arbol 1 fila 5
            n_p = os.path.join('../Revisado', 'arbol_' + d.split('_')[0] + '_fila_' + d.split('_')[1]+'.JPG')
            print(n_p)
            shutil.copy(op, n_p)
        myfile.close()
    
    def label(self):
        base = os.path.split(self.path)[-1]
        try:
            os.mkdir(os.path.join(os.pardir, 'Labeled1'))
            
        except:
            pass
        os.mkdir(os.path.join(os.pardir, 'Labeled1', base))
        self.get_2dcoordinates()
        myfile = open('unlabeled.txt', 'w')
        for i in range(self.plane.shape[0]):
            #print(str(i) + "\n")
            flag = False
            #if i == 30:
            #    break
            for j in self.rd.keys():
                dd = (self.plane[i][0]-self.rd[j][0])**2 + (self.plane[i][1]-self.rd[j][1])**2
                #print(dd)
                if(dd < 0.008):
                    copy2(self.__getitem__(i)[2], os.path.join(os.pardir, "Labeled1",base,base+"__"+str(j)+".JPG"))
                    flag = True
                    break
            if not flag:
                myfile.write("%s\n" % os.path.split(self.__getitem__(i)[2])[-1])
                    



            

if __name__ == '__main__':
    #lab = img_set(path = '../5_agosto')
    #itp = os.path.join(os.pardir, "Data_Base")
    itp = "/home/liiarpi-01/Downloads/Phantom_LASTS"
    for fol in os.listdir(itp):
        lab = img_set(path=os.path.join(itp, fol))
        lab.label()
    #lab = img_set(path = '../unlabeled_images')
    #lab.plot(0)
    #lab.plot(1)
    #lab.plot(1)
    
    print(len(lab))


