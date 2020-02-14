# -*- coding: utf-8 -*-
import torch
from torch.utils.data import Dataset


class WholeDataSet(Dataset):
    def __init__(self,X,y,d):
        self.X,self.y,self.d = torch.from_numpy(X),torch.from_numpy(y),torch.from_numpy(d)

    def __getitem__(self,index):
        return self.X[index,:],self.y[index,None],self.d[index,None]

    def __len__(self):
        return self.X.shape[0]



