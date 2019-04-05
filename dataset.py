#!/usr/bin/env
# -*- coding: utf-8 -*-
''' 
Copyright (c) 2019 Sufer_Qin
'''

# data process
import numpy as np
import pandas as pd
from PIL import Image

# For network and training
import torch
import torch.nn as nn
import torch.nn.functional as F

# For dataset
import torchvision
import torchvision.transforms as T

# ------------------------------------------ Datatset ------------------------------------------
# Define dataset class
class cifar10_Dataset(torch.utils.data.Dataset):
    def __init__(self, dataType, Num, transforms = None,has_label = True, **kwargs):
        super(cifar10_Dataset, self).__init__()
        if Num is None:
            data = pd.read_csv('.\\input\\'+dataType+'.csv')
            self.lens = len(data)
        else:
            data = pd.read_csv('.\\input\\'+dataType+'.csv', nrows=Num)
            self.lens = Num
        if has_label:
            self.labels = torch.from_numpy(data['Category'].values).long()
            self.imgs = torch.from_numpy(data.drop(['ID', 'Category'], axis=1).values.reshape(-1, 3, 32, 32)).float()/255.0
        else:
            self.labels = torch.from_numpy(data['ID'].values).long()
            self.imgs = torch.from_numpy(data.drop(['ID'], axis=1).values.reshape(-1, 3, 32, 32)).float()/255.0
        self.transforms = transforms

    def __getitem__(self, index):
        return self.transforms(self.imgs[index, :, :, :]), self.labels[index]

    def __len__(self):
        return self.lens

def loadData(name, Num, batch):
    if name.lower() =="train":
        transforms = T.Compose([
            T.ToPILImage(),
            T.RandomResizedCrop(32,(0.7,1),(0.75,1.33)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        dataset = cifar10_Dataset(name,Num,transforms)

    elif name == 'validation':
        transforms = T.Compose([
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        dataset = cifar10_Dataset('test',Num,transforms)

    elif name == "TEST" :
        transforms = T.Compose([
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        dataset = cifar10_Dataset(name,Num,transforms,False)

    return torch.utils.data.DataLoader(dataset, batch_size=batch, shuffle=True, num_workers=2)

if __name__ == '__main__':
    pass