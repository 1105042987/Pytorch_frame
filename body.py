#!/usr/bin/env
# -*- coding: utf-8 -*-
''' 
Copyright (c) 2019 Charles, Sufer_Qin
'''

import numpy as np
from torchnet import meter  # Easy for record loss
import time
from tqdm import tqdm
import os
from PIL import Image
import pandas as pd
import argparse
from datetime import datetime
from abc import ABCMeta, abstractmethod
isExist = os.path.exists


# For network and training
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse

# For dataset
import torchvision
import torchvision.transforms as T



loss_dic={
    'crossE':nn.CrossEntropyLoss(),
    'mae':nn.L1Loss(),
    'mse':nn.MSELoss(),
    'bce':nn.BCEWithLogitsLoss(),
}

def base_args():
    from train import net_dic
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('-n', nargs='?', type=str, choices=list(net_dic.keys()), default = list(net_dic.keys())[0],
                        help="Net choose: {}".format(list(net_dic.keys())))

    parser.add_argument('--loss', nargs='?', type=str, choices=list(loss_dic.keys()),default=list(loss_dic.keys())[0],
                        help="Net choose: {}".format(list(loss_dic.keys())))

    parser.add_argument('--suf', nargs='?', type=str, default = '',
                        help='Suffix name')

    parser.add_argument('--opt', nargs='?', type=str, choices=['adam','sgd'], default = 'adam',
                        help="Optimizer type: ['adam','sgd'],default 'adam'")

    parser.add_argument('--max_iter', nargs='?', type=int, default = 300,
                        help='Epoch num, default 300')

    parser.add_argument('--lr_step', nargs='?', type=int, default = 50,
                        help="Every 'lr_step' epoch, 'lr' decay , default 50")

    parser.add_argument('--lr_decay', nargs='?', type=int, default = 0.6,
                        help="Learning rate decay rate, default 0.6")

    parser.add_argument('--lr', nargs='?', type=float, default=0.001,
                        help='Learning rate, default 0.001 (1e-3)')

    parser.add_argument('--weight_decay', nargs='?', type=float, default=0.0001,
                        help='Weight_decay, default 0.0001 (1e-4)')

    parser.add_argument('--gpu', nargs='?', type=int, default=0,
                        help='GPU num')

    parser.add_argument('--batch', nargs='?', type=int, default=128,
                        help='Batch size')

    parser.add_argument('--resume', nargs='?', type=str, default = None,
                        help='Continue last training with the address you give')
    return parser


class weak_ErrorRate(metaclass=ABCMeta):
    def __init__(self):
        super(weak_ErrorRate,self).__init__()
        self.reset()

    def reset(self):
        self.total = 0
        self.error = 0

    def value(self):
        return [self.error/self.total]

    @abstractmethod
    def add(self,out,tar):
        pass



class weak_GetPredictAnswer(metaclass=ABCMeta):
    def __init__(self):
        super(weak_GetPredictAnswer,self).__init__()
        self.data = None

    @abstractmethod
    def add(self,outputs,index):
        pass

    @abstractmethod
    def save(self,name):
        pass
        


# ---------------------------------- Network Architecture ----------------------------------
class Body(object):
    def __init__(self,args):
        super(Body,self).__init__()
        timestamp = datetime.now().strftime('%m%d_%H[%M]')
        self.device = torch.device('cuda', args.gpu) if torch.cuda.is_available() else torch.device('cpu')
        
        if args.resume is not None:
            self.name = args.resume[14:-4]
            self.resume(net_dic)
        else:
            self.name = timestamp+'_'+args.n if args.suf=='' else self.timestamp+'_'+args.n+'_'+args.suf
            if isExist('.\\checkpoints\\'+self.name+'.pth'):
                raise('[{}] has been used!'.format(self.name))

            self.args = args
            self.best = 1000
            self.start = 0
            self.build_model()

        
    def resume(self):
        if isExist('.\\checkpoints\\'+self.name+'.pth'):
            state = torch.load('.\\checkpoints\\'+self.name+'.pth',map_location=lambda storage, loc: storage)

            self.args = state['args']
            self.best = state['best']
            self.start = state['epoch']
            self.build_model()

            self.net.load_state_dict(state['net'])
            self.opt.load_state_dict(state['opt'])
            self.sch.load_state_dict(state['sch'])
        else:
            raise('[{}] do not exist!'.format(self.name))


    def build_model(self):
        # Must be used after self have 'args' and 'device'
        from train import refresh_net_dic, net_dic
        refresh_net_dic(net_dic,self.args)
        self.net = net_dic[self.args.n].to(self.device)
        if self.args.opt == 'adam':
            self.opt = torch.optim.Adam(self.net.parameters(), lr = self.args.lr, weight_decay = self.args.weight_decay)
        elif self.args.opt == 'sgd':
            self.opt = torch.optim.SGD(self.net.parameters(), lr = self.args.lr, weight_decay = self.args.weight_decay)
        self.sch = torch.optim.lr_scheduler.StepLR(self.opt, step_size = self.args.lr_step, gamma = self.args.lr_decay)
        self.criterion = loss_dic[self.args.loss]


    def save(self,epoch,compare):
        if compare<self.best:
            self.best = compare
            torch.save({
                'best': self.best,
                'epoch':epoch,
                'args':self.args.__dict__,
                'net': self.net.state_dict(),
                'opt' : self.opt.state_dict(),
                'sch' : self.sch.state_dict(),
            }, '.\\checkpoints\\'+self.name+'.pth')


    def log_record(self,file,epoch,print_dic):
        if epoch is None:
            log = "[Test]:  "
        else:
            log = 'Epoch {}, '.format(epoch)

        for key,value in print_dic.items():
            if value is not None:
                log += '{}: {}, '.format(key,value.value()[0])

        print(log)

        if file is not None:
            file.write(log+'\n')
        


    def train(self,train_loader,validation_loader=None,ErrorClass=weak_ErrorRate):
        if not issubclass(ErrorClass,weak_ErrorRate):
            raise("'ErrorClass' is not the sub class of 'weak_ErrorRate'")
        with open('.\\logs\\'+self.name+'.log', 'a') as file:
            # create record meter
            loss_meter= meter.AverageValueMeter()
            error_meter = ErrorClass() if ErrorClass is not weak_ErrorRate else None
            if validation_loader is not None:
                loss_meter2 = meter.AverageValueMeter()
                error_meter2 = ErrorClass() if ErrorClass is not weak_ErrorRate else None
            else:
                loss_meter2,error_meter2 = None,None
            print_dic={
                'Train_Loss':loss_meter,
                'Train_Error':error_meter,
                'Validation_Loss':loss_meter2,
                'Validation_Error':error_meter2,
            }
            for epoch in range(self.start,self.args.max_iter):
                # Make sure you enable the auto-grad engine
                self.net.train()
                loss_meter.reset()
                error_meter.reset()
                # train
                for inputs, targets in tqdm(train_loader):
                    inputs, targets = inputs.to(self.device), targets.to(self.device)

                    self.opt.zero_grad()
                    outputs = self.net(inputs)
                    loss = self.criterion(outputs, targets)

                    loss.backward()
                    self.opt.step()

                    loss_meter.add(loss.item())
                    if error_meter is not None:
                        error_meter.add(outputs,targets)
                self.sch.step()
                # validation
                if validation_loader is not None:
                    # Make sure you disable the auto-grad engine
                    self.net.eval()
                    loss_meter2.reset()
                    error_meter2.reset()
                    for inputs, targets in tqdm(validation_loader):
                        inputs, targets = inputs.to(self.device), targets.to(self.device)

                        outputs = self.net(inputs)
                        loss = self.criterion(outputs, targets)

                        loss_meter2.add(loss.item())
                        if error_meter2 is not None:
                            error_meter2.add(outputs,targets)
                # record
                self.log_record(file,epoch,print_dic)
                self.save(epoch,loss_meter.value()[0])


    def evaluate(self,test_loader,ErrorClass=weak_ErrorRate):
        if not issubclass(ErrorClass,weak_ErrorRate):
            raise("'ErrorClass' is not the sub class of 'weak_ErrorRate'")
        # Cancel auto-grad engine
        self.net.eval() 
        loss_meter = meter.AverageValueMeter()
        error_meter = ErrorClass() if ErrorClass is not weak_ErrorRate else None
        print_dic={
            'Test_Loss':loss_meter,
            'Test_Error':error_meter,
        }
        loss_meter.reset()
        error_meter.reset()
        for inputs, targets in tqdm(test_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            outputs = self.net(inputs)
            loss = self.criterion(outputs, targets)
           
            loss_meter.add(loss.item())
            error_meter.add(outputs,targets)

        self.log_record(None,None,print_dic)
        


    def predict(self,test_loader,AnsClass = weak_GetPredictAnswer):
        if not issubclass(AnsClass,weak_GetPredictAnswer):
            raise("'AnsClass' is not the sub class of 'weak_GetPredictAnswer'")
        if AnsClass is weak_GetPredictAnswer:
            raise("'AnsClass' has abstract method")
        answer = AnsClass()
        # Cancel auto-grad engine
        self.net.eval()  
        for inputs,index in tqdm(test_loader):
            inputs = inputs.to(self.device)
            answer.add(self.net(inputs),index)
        answer.save(self.name)


def num2onehot(num_label,scatter_target_num):
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



