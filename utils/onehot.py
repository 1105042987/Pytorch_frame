#!/usr/bin/env
# -*- coding: utf-8 -*-
''' 
Copyright (c) 2019 Sufer_Qin
'''
import torch

def num2onehot(num_label, scatter_target_num):
    if type(num_label) != torch.Tensor:
        raise('num2onehot: num\'s type must be torch.Tensor')
    if len(num_label.shape) != 2 or num_label.shape[1] != 1:
        raise('num2onehot: num must in shape (-1,1)')
    return torch.zeros(num_label.shape[0], scatter_target_num).scatter_(1, num_label, 1).long()


def onehot2num(onehot_label):
    if type(onehot_label) != torch.Tensor:
        raise('onehot2num: onehot\'s type must be torch.Tensor')
    if len(onehot_label.shape) != 2:
        raise('onehot2num: onehot must in shape (len_of_batch,len_of_target_class)')
    return torch.max(onehot_label, 1)[1].reshape(-1, 1).long()
