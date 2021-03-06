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
class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataType, Num, transforms = None,has_label = True, **kwargs):
        super(Dataset, self).__init__()
        if Num is None:
            data = pd.read_csv('./dataset/'+dataType+'.csv')
            self.lens = len(data)
        else:
            data = pd.read_csv('./dataset/'+dataType+'.csv', nrows=Num)
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
    if name.lower() == "train":
        transforms = T.Compose([
            T.ToPILImage(),
            T.RandomCrop(32, padding=4),
            # T.RandomResizedCrop(32,(0.7,1),(0.75,1.33)),
            T.RandomHorizontalFlip(),
            # T.RandomVerticalFlip(),
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        dataset = Dataset(name,Num,transforms)

    elif name == 'validation':
        transforms = T.Compose([
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        dataset = Dataset('test',Num,transforms)

    elif name == "TEST" :
        transforms = T.Compose([
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        dataset = Dataset(name,Num,transforms,False)

    return torch.utils.data.DataLoader(dataset, batch_size=batch, shuffle=True, num_workers=2)

from frame.frame import weak_ErrorRate,weak_GetPredictAnswer
from utils.onehot import onehot2num

class ErrorRate(weak_ErrorRate):
    def __init__(self):
        super(ErrorRate, self).__init__()

    def add(self, out, tar):
        import torch
        self.total += len(tar)
        outputs = torch.max(out, 1)[1]
        self.error += torch.sum(outputs != tar).float()


class AnsGet(weak_GetPredictAnswer):
    def __init__(self):
        super(AnsGet, self).__init__()

    def add(self, outputs, index):
        from torch import cat
        index=index.reshape(-1, 1)
        if self.data is None:
            self.data = cat((index, onehot2num(outputs).to('cpu')), 1)
        else:
            outputs = cat((index, onehot2num(outputs).to('cpu')), 1)
            self.data = cat((self.data, outputs))

    def save(self, name):
        import pandas as pd
        ans = pd.DataFrame(np.array(self.data), columns=['ID', 'Category'])
        ans.to_csv('./output/{}.csv'.format(name), header=True, index=False)


if __name__ == '__main__':
    pass 
    # import pandas as pd
    # import numpy as np
    # pic = pd.read_csv('train.csv', nrows=3)
    # pic1 = pic.drop(['ID', 'Category'], axis=1).values.reshape(-1, 3, 32, 32)/255.
    # pic2 = pic.drop(['ID', 'Category'], axis=1).values.reshape(-1,32, 32,3)/255.
    # print(pic1.shape)
    # import cv2
    # cv2.imshow('3xx', pic1[0, 0, :, :])
    # cv2.imshow('xx3', pic2[0, :, :, 0])
    # cv2.waitKey(0)
