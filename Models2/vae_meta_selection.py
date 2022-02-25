import numpy as np
import pandas as pd
from utils import *
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from copy import copy
import torch.optim as optim
from tqdm import tqdm
df = pd.read_csv('C:\\Users\\LENOVO\\Downloads\\gmvae_a1_2.csv', index_col=0)
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
N_data = df[(df['Class']=='N_Deficiencia') | (df['Class']=='N_Control') | (df['Class']=='N_Exceso')]
print(N_data.Class.unique())
#N_data = N_data.iloc[:,:-1]
n2clas={'N_Deficiencia':0, 'N_Control':1, 'N_Exceso':2}
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
    for fold, (train_ids, test_ids) in enumerate(skf.split(N_data, N_data.iloc[:,-1])):
        l_v = []
        a_v = []
        f_v = []
        #print(f'Fold {fold+1}')
        #print('------------------------')
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
        train_loader = torch.utils.data.DataLoader(
                          a, 
                          batch_size=512, sampler=train_subsampler)
        test_loader = torch.utils.data.DataLoader(
                          a,
                          batch_size=1, sampler=test_subsampler)
        model = SimpleClassification(n_features=n_f)
        model.to(device)
        #print(model)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr = 0.001)
        itt = tqdm(range(EPOCHS))
        for i in itt:
            loss_epoch = 0
            acc_train, acc_test = 0.0, 0.0
            f1_train, f1_test = 0.0, 0.0
            model.train()
            for X, y in train_loader:
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

                for X, y in test_loader:
                    X, y = X.to(device), y.to(device).flatten()
                    model.eval()
                    output = model(X)
                    acc = multi_acc(output, y)
                    f1 = F1_score(output, y)
                    acc_test += acc.item()
                    f1_test += f1.item()

            itt.set_description("Acc train: %.2f Acc test: %.2f F1 train: %.2f F1 test: %.2f" % (acc_train/len(train_loader), acc_test/len(test_loader), f1_train/len(train_loader), f1_test/len(test_loader)))

            l_v.append(loss_epoch)
            a_v.append(acc_test/len(test_loader))
            f_v.append(f1_test/len(test_loader))
            
            #print(f'Loss {i+1}: {loss_epoch}')
            #print(f'Acc {i+1}: {acc_epoch/len(train_loader)}')
            #print(f'F1 {i+1}: {f1_epoch/len(train_loader)}')
        #print(a_v, f_v)
        aux_f1_idx.append(f_v)
        aux_ac_idx.append(a_v)
        
    aux_ac_idx = np.array(aux_ac_idx)
    aux_f1_idx = np.array(aux_f1_idx)
    #print(aux_ac_idx, aux_f1_idx)
    idx_result['f1'] = max(np.mean(aux_f1_idx, axis = 0))
    idx_result['acc'] = max(np.mean(aux_ac_idx, axis = 0))
    return idx_result
        #plt.figure(), plt.plot(l_v), plt.title('Loss by epoch')
        #plt.figure(), plt.plot(a_v), plt.title('Accuracy test by epoch')
        #plt.figure(), plt.plot(f_v), plt.title('F1 test by epoch')
        #plt.show()
    
N_FEATURES = 33



###SELECTOR
def select_k_best(k = 10, ev = 'f1'):
    """
    ev = 'acc' or 'f1'
    """
    
    
    list_features = list(range(N_FEATURES))
    initial_feat = []
    while(len(initial_feat)<k):
        selector = {}
        for i in list_features:
            initial_feat.append(i)
            print(f"=======================================Using {initial_feat} ===============")
            res = train(initial_feat, ep = 100)
            initial_feat.pop()
            selector.update({i:res[ev]})
            with open('nit_gm_vae.txt', mode='a') as f:
                f.write(str(res))
                f.write('\n')

        best_feature = max(selector, key=selector.get)
        initial_feat.append(best_feature)
        list_features.remove(best_feature)
        print("================================================")
        print(f"=================ADEDD {max(selector, key=selector.get)} TO BEST FEATURES ==========")
        print(f"================={ev} = {selector[best_feature]} ================================")

    
    print(f"FINAL FEATURES")
    print(initial_feat)
    

select_k_best(k = 30)

