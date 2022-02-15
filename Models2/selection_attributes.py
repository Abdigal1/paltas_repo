import os
import pickle
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.functional as F
from sklearn.preprocessing import MinMaxScaler
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from meta_load import *
from copy import copy

##CARGA DATOS
#df = load_meta_v2('C:\\Users\\LENOVO\\Desktop\\NIRLBCM')
df = load_meta_v2('C:\\Users\\LENOVO\\Desktop\\NIRLBCM')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

##SELECCION DE N Y CLASES
N_data = df[(df['landmark']=='P_Control') | (df['landmark']=='P_Exceso') | (df['landmark']=='P_Deficiencia')]
#optional
#N_data = N_data.iloc[:,[4,5,10,11,16,17,22,23,28,29,30,31,32]]
N_data = N_data[N_data['fly']!=0]

n2clas={'P_Deficiencia':0, 'P_Control':1, 'P_Exceso':2}
N_data.landmark.replace(n2clas, inplace=True)

##DATASET CLASE
class datapaltas(Dataset):
    def __init__(self, df=N_data, scale =True):
        self.df = df

    def __getitem__(self, index):
        X = (self.df.iloc[index,:-2]).values
        X = X.astype(np.float64)
        X = torch.from_numpy(X).float()
        y = self.df.iloc[index,-2]
        #print(y)
        y = torch.Tensor([y]).long()
        return X, y
    
    def __len__(self):
        return self.df.shape[0]
##MODELO

class SimpleClassification(nn.Module):
    
    def __init__(self, n_features = 11, n_classes = 3):
        super(SimpleClassification, self).__init__()
        self.n_features = n_features
        self.l1 = nn.Linear(self.n_features, 128)
        self.l2 = nn.Linear(128, 64)
        self.l3 = nn.Linear(64, 32)
        self.l4 = nn.Linear(32, 16)
        self.l5 = nn.Linear(16, n_classes)
        
        self.relu = nn.ReLU()
        self.batchnorm1 = nn.BatchNorm1d(128)
        self.batchnorm2 = nn.BatchNorm1d(64)
        self.batchnorm3 = nn.BatchNorm1d(32)
        self.batchnorm4 = nn.BatchNorm1d(16)
    
    def forward(self, x):
        x = self.relu(self.batchnorm1(self.l1(x)))
        x = self.relu(self.batchnorm2(self.l2(x)))
        x = self.relu(self.batchnorm3(self.l3(x)))
        x = self.relu(self.batchnorm4(self.l4(x)))
        return self.l5(x)

##FUNCIONES PARA CALIFICAR EL MODELO
def multi_acc(y_pred, y_test):
    y_pred_softmax = torch.log_softmax(y_pred, dim = 1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim = 1)    
    
    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)
    
    acc = torch.round(acc * 100)
    
    return acc

def F1_score(prob, label):
    y_pred_softmax = torch.log_softmax(prob, dim = 1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim = 1)
    #print(y_pred_tags, label)
    return f1_score(label.cpu(), y_pred_tags.cpu(), average='macro')





#####MODELO PARA ENTRENAR ESPECIFICANDO CARACTERISITICAS
def train(idx, ep = 100):
    idx_result = {'idx_used':copy(idx), 'f1': 0, 'acc':0}
    idx_copy = copy(idx)
    aux_f1_idx = []
    aux_ac_idx = []
    n_f = len(idx_copy)
    idx_copy.extend([-2, -1])
    a = datapaltas(df = N_data.iloc[:, idx_copy])
    EPOCHS = ep
    
    skf = StratifiedKFold()
    skf.get_n_splits(N_data, N_data.iloc[:,-2])
    for fold, (train_ids, test_ids) in enumerate(skf.split(N_data, N_data.iloc[:,-2])):
        l_v = []
        a_v = []
        f_v = []
        #print(f'Fold {fold+1}')
        #print('------------------------')
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
        train_loader = torch.utils.data.DataLoader(
                          a, 
                          batch_size=64, sampler=train_subsampler)
        test_loader = torch.utils.data.DataLoader(
                          a,
                          batch_size=1, sampler=test_subsampler)
        model = SimpleClassification(n_features=n_f)
        model.to(device)
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
    
N_FEATURES = 31

###SELECTOR
def select_k_best(k = 10, ev = 'f1'):
    """
    ev = 'acc' or 'f1'
    """
    total_log = []
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
            total_log.append(res)
        best_feature = max(selector, key=selector.get)
        initial_feat.append(best_feature)
        list_features.remove(best_feature)
        print("================================================")
        print(f"=================ADEDD {max(selector, key=selector.get)} TO BEST FEATURES ==========")
        print(f"================={ev} = {selector[best_feature]} ================================")
        
    print(f"FINAL FEATURES")
    print(initial_feat)
    return total_log

log = select_k_best(k = 10)

with open('log.txt', mode='w') as f:
    f.write(str(log))