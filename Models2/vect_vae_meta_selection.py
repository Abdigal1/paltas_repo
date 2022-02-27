import numpy as np
import pandas as pd
from utils import *
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from copy import copy
import torch.optim as optim
from tqdm import tqdm




def avr(train_ids, test_ids, n_f, ep, a):
    l_v = np.zeros((ep,))
    a_v = np.zeros((ep,))
    f_v = np.zeros((ep,))
    #print(f'Fold {fold+1}')
    #print('------------------------')
    #print(a[0].dtype)
    #print(train_ids.shape)
    #print(test_ids.shape)
    train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
    test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
    train_loader = torch.utils.data.DataLoader(
                        a, 
                        batch_size=256, sampler=train_subsampler)
    test_loader = torch.utils.data.DataLoader(
                        a,
                        batch_size=1, sampler=test_subsampler)
    #print(type(train_loader), type(test_loader))
    model = SimpleClassification(n_features=n_f)
    model.to(device)
    #print(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = 0.001)
    EPOCHS = ep
    itt = tqdm(range(EPOCHS))
    for i in itt:
        loss_epoch = 0
        acc_train, acc_test = 0.0, 0.0
        f1_train, f1_test = 0.0, 0.0
        model.train()
        for sample in train_loader:
            X, y = sample['x'], sample['y']
            X, y = X.to(device), y.to(device).flatten()
            optimizer.zero_grad()
            y_pred = model(X)
            #print(y_pred, y)
            loss = criterion(y_pred, y)
            acc = multi_acc(y_pred, y)
            f1 = F1_score(y_pred, y)
            loss.backward()
            optimizer.step()
            loss_epoch += loss.item()
            acc_train += acc.item()
            f1_train += f1.item()

        with torch.no_grad():

            for sample in test_loader:
                X, y = sample['x'], sample['y']
                X, y = X.to(device), y.to(device).flatten()
                model.eval()
                output = model(X)
                acc = multi_acc(output, y)
                f1 = F1_score(output, y)
                acc_test += acc.item()
                f1_test += f1.item()

            l_v[i] = loss_epoch
            a_v[i] = acc_test/len(test_loader)
            f_v[i] = f1_test/len(test_loader)
        itt.set_description("Acc train: %.2f Acc test: %.2f F1 train: %.2f F1 test: %.2f" % (acc_train/len(train_loader), acc_test/len(test_loader), f1_train/len(train_loader), f1_test/len(test_loader)))

    return l_v, a_v, f_v

avr_v = np.vectorize(avr,signature="(i),(j),(),(),(k)->(a),(b),(c)")





df = pd.read_csv('C:\\Users\\LENOVO\\Downloads\\\\GMVAE_A2_1.csv', index_col=0)
df['Date']=[i[:-2] for i in df['Date'].values]
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
flies_dict = {j:i for i, j in enumerate(a)}
df.Date.replace(flies_dict, inplace=True)
#df.iloc[:,-1], df.iloc[:,-2] = df.iloc[:,-2], df.iloc[:,-1]
#colist = list(df)
#colist[-1], colist[-2] = colist[-2], colist[-1]
#df.columns = colist
device = 'cpu'
##SELECCION DE N Y CLASES
N_data = df[(df['Class']=='H50%') | (df['Class']=='H75%') | (df['Class']=='Control')]
N_data = N_data.iloc[:-1, :]
print(N_data.Class.unique())
#N_data = N_data.iloc[:,:-1]
n2clas={'H50%':0, 'H75%':1, 'Control':2}
N_data.Class.replace(n2clas, inplace=True)
print(N_data.Class.unique())
#print(N_data)
def train(idx, ep = 100):
    idx_result = {'idx_used':copy(idx), 'f1': 0, 'acc':0}
    idx_copy = copy(idx)
    aux_f1_idx = []
    aux_ac_idx = []
    n_f = len(idx_copy)
    idx_copy.extend([-1])
    a = datapaltas(df = N_data.iloc[:, idx_copy], y_idx=-1)
    EPOCHS = ep
    
    skf = StratifiedKFold()
    ## y_idx -2
    skf.get_n_splits(N_data, N_data.iloc[:,-1])
    vec_train = np.array([train_ids for train_ids,_ in skf.split(N_data, N_data.iloc[:,-1])])
    vec_test = np.array([test_ids for _,test_ids in skf.split(N_data, N_data.iloc[:,-1])])
    #for i, j in zip(vec_train, vec_test):
    #    print(len(i), len(j))
    
    
    #print(avr(vec_train[0], vec_test[0], n_f, ep))
    #print(vec_train.shape, vec_test.shape)
    #print(f"AVER {n_f}")
    lvv, acv, f1v = avr_v(vec_train, vec_test, n_f, ep, a)
    
    #print(aux_ac_idx, aux_f1_idx)
    idx_result['f1'] = max(np.mean(f1v, axis = 0))
    idx_result['acc'] = max(np.mean(acv, axis = 0))
    return idx_result
        #plt.figure(), plt.plot(l_v), plt.title('Loss by epoch')
        #plt.figure(), plt.plot(a_v), plt.title('Accuracy test by epoch')
        #plt.figure(), plt.plot(f_v), plt.title('F1 test by epoch')
        #plt.show()
    
N_FEATURES = 45



###SELECTOR
def select_k_best(k = 10, ev = 'f1'):
    """
    ev = 'acc' or 'f1'
    """
    
    
    list_features = list(range(N_FEATURES))
    initial_feat = list(range(N_FEATURES))
    print(len(initial_feat), k)
    print(len(initial_feat)>k)
    while len(initial_feat)>k:
        selector = {}
        for i in list_features:
            initial_feat.remove(i)
            print(f"=======================================Using {initial_feat} ===============")
            res = train(initial_feat, ep = 100)
            initial_feat.append(i)
            selector.update({i:res[ev]})
            with open('ra.txt', mode='a') as f:
                f.write(str(res))
                f.write('\n')

        worst_feature = min(selector, key=selector.get)
        initial_feat.remove(worst_feature)
        list_features.remove(worst_feature)
        print("================================================")
        print(f"=================REMOVED {min(selector, key=selector.get)} FROM FEATURES ==========")
        print(f"================={ev} = {selector[worst_feature]} ================================")

    
    print(f"FINAL FEATURES")
    print(initial_feat)
    

select_k_best(k = 10)

