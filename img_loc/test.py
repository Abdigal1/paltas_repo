import os
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
import matplotlib.pyplot as plt
import numpy as np

#print(TAGS)
#print(GPSTAGS)

def getim_meta(path):

    img = Image.open(path)
    info = img._getexif()
    GPSDATA = info[list(TAGS.keys())[list(TAGS.values()).index('GPSInfo')]]
    #Latitud, Longitud, Altitud
    return (GPSDATA[2][2], GPSDATA[4][2], GPSDATA[6])

#PATH_TEST = '..\\labeled_images'
PATH_TEST = '..\\unlabeled_images'
X = os.listdir(PATH_TEST)
val = map(lambda X:getim_meta(os.path.join(PATH_TEST, X)), X)
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
"""

miny = np.min(plane[:, 1])
plane[:, 1] = plane[:, 1]-miny
MAX_Y = np.ceil(np.max(plane[:, 1]))

plane = plane + 20
img = np.zeros((MAX_X+40, MAX_Y+40), dtype=np.uint8)
"""
