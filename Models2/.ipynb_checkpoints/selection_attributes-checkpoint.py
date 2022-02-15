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

class datapaltas(Dataset):
    def __init__(self, df, scale =True):
        self.df = df

    def __getitem__(self, index):
        X = (self.df.iloc[index,:-3]).values
        X = X.astype(np.float64)
        X = torch.from_numpy(X).float()
        y = self.df.iloc[index,11]
        #print(y)
        y = torch.Tensor([y]).long()
        return X, y
    
    def __len__(self):
        return self.df.shape[0]

class SimpleClassification(nn.Module):
    
    def __init__(self, n_features = 30, n_classes = 3):
        super(SimpleClassification, self).__init__()
        
        self.l1 = nn.Linear(n_features, 128)
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

df = load_meta_v2('C:\\Users\\abdig\\Desktop\\NIRLBCM')
device = 'cuda' if torch.cuda.is_available() else 'cpu'


N_data = df[(df['landmark']=='N_Control') | (df['landmark']=='N_Exceso') | (df['landmark']=='N_Deficiencia')]
#optional
#N_data = N_data.iloc[:,[4,5,10,11,16,17,22,23,28,29,30,31,32]]
N_data = N_data[N_data['fly']!=0]

n2clas={'N_Deficiencia':0, 'N_Control':1, 'N_Exceso':2}

N_data.landmark.replace(n2clas, inplace=True)

a = datapaltas(df=N_data)
print(a[0])
train_loader = DataLoader(dataset=a, batch_size=64)
model = SimpleClassification(n_features= 30)
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.001)


EPOCHS = 1

skf = StratifiedKFold()
skf.get_n_splits(N_data, N_data['landmark'])
for fold, (train_ids, test_ids) in enumerate(skf.split(N_data, N_data['landmark'])):
    l_v = []
    a_v = []
    f_v = []
    print(f'Fold {fold+1}')
    print('------------------------')
    train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
    test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
    train_loader = torch.utils.data.DataLoader(
                      a, 
                      batch_size=64, sampler=train_subsampler)
    test_loader = torch.utils.data.DataLoader(
                      a,
                      batch_size=1, sampler=test_subsampler)
    model = SimpleClassification()
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = 0.001)
    itt = tqdm(range(EPOCHS))
    for i in itt:
        loss_epoch = 0
        acc_train, acc_test = 0, 0
        f1_train, f1_test = 0, 0
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
    plt.figure(), plt.plot(l_v)
    plt.figure(), plt.plot(a_v)
    plt.figure(), plt.plot(f_v)
    plt.show()