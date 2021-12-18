import os
import numpy as np
import pickle
import pandas as pd
a = ['29_marzo',
 '14_abril',
 '28_abril',
 '7_mayo',
 '12_mayo',
 '19_mayo',
 '26_mayo',
 '2_junio',
 '11_junio',
 '16_junio',
 '23_junio',
 '2_julio',
 '9_julio',
 '14_julio',
 '23_julio',
 '5_agosto',
 '13_agosto',
 '19_agosto',
 '15_setiembre',
 '24_setiembre',
 '15_octubre',
 '29_octubre',
 '12_noviembre',
 '26_noviembre']

met_val = ['15_0' ,'15_90', '30_0' ,'30_90', '50_0' ,'50_90']
flies_dict = {j:i for i, j in enumerate(a)}
def load_df(file):
    out = []
    with open(file, 'rb') as handle:
        try:
            while True:
                out.append(pickle.load(handle))
        except EOFError:
            pass
    return out


PATHG = os.path.join(os.pardir, 'Data_prep')
def load_meta_v2():
    apkl = []
    for i in os.listdir(PATHG):
        
        if i.endswith('2.pkl'):
            apkl.append(load_df(os.path.join(PATHG,i)))
        
    sa = []
    fly = []
    place = []
    landmark = []
    print(len(apkl), len(apkl[0]))
    for i in range(len(apkl[0])):
        auxv = []
        for j in range(len(apkl)):
            key = list(apkl[j][i].keys())[0]
            if j==0:
                fly.append(flies_dict[(apkl[j][i]['Date'])[:-2]])
                landmark.append(apkl[j][i]['landmarks'])
                place.append(apkl[j][i]['Place'])
            
            auxv.extend(apkl[j][i][key].flatten())
        sa.append(auxv)
    sa = np.array(sa)
    fly = np.array(fly)
    place = np.array(place)
    landmark = np.array(landmark)

    print(sa.shape, fly.shape, place.shape, landmark.shape)
    out = {}
    for i in range(len(apkl)):
        key = list(apkl[i][0].keys())[0]
        for j, item in enumerate(met_val):
            out.update({key+item:sa[:,6*i+j]})

    out.update({'fly':fly, 'landmark':landmark, 'Place':place})
    return pd.DataFrame(out)


