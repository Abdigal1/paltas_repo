import numpy as np
import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score

class datapaltas(Dataset):
    def __init__(self, df, scale =True, y_idx=-2):
        self.df = df
        self.y_idx = y_idx

    def __getitem__(self, index):
        X = (self.df.iloc[index,:self.y_idx]).values
        X = X.astype(np.float64)
        X = torch.from_numpy(X).float()
        y = self.df.iloc[index,self.y_idx]
        #print(y)
        y = torch.Tensor([y]).long()
        return {'x':X, 'y':y}
    
    def __len__(self):
        return self.df.shape[0]






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