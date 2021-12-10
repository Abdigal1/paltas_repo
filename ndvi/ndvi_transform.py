from os import stat
from utils import *
import matplotlib.pyplot as plt

class ndvi_desc():
    """Compute NDVI from RGB and NIR image 
    
    Args:
        statistic(boolean): Return either mean and std
        or masked NDVI image
        
    output:
        NDVI mean and std

    """

    def __init__(self, statistic_ = False):
        self.statistic = statistic_

    def __call__(self, sample):
        rgb, nir, mask, landmarks = sample['SenteraRGB'],\
            sample['SenteraNIR'], sample['SenteraMask'],\
                 sample['landmarks']
        
        if self.statistic:
            mean, std = compute_ndvi(rgb,nir, mask)
            return{'SenteraNDVI': {'mean': mean, 'std': std}, 'landmarks': landmarks}
        else:
            ndvi = compute_ndvi(rgb,nir, mask, statistic=False)
            return{'SenteraNDVI': ndvi, 'landmarks': landmarks}
            



#if __name__ == '__main__':
#    a = cv2.imread('..\\Tests\\tela_RGB7.jpg')
#    b = cv2.imread('..\\Tests\\tela_NIR7.jpg')
#    #p, q, r = get_clothe_masks(a)
#    #plt.imshow(compute_ndvi(a,b, r, statistic=False))
#    #plt.show()
#    print([compute_ndvi(a,b, i) for i in get_clothe_masks(a)])
        



