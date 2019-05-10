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
from re import findall
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

def base_args(omit = False):
    parser = argparse.ArgumentParser(description='PyTorch Training')

    parser.add_argument('-gpu', nargs='?', type=int, default=0,
                        help="GPU number,    Default 0")
    if omit == False:
        from train import net_dic
        parser.add_argument('-n', nargs='?', type=str, choices=list(net_dic.keys()), default = list(net_dic.keys())[0],
                            help="Net choose:{}, Default:{}".format(list(net_dic.keys()),list(net_dic.keys())[0]))

        parser.add_argument('-suf', nargs='?', type=str, default = '',
                            help="Suffix name")

        parser.add_argument('-max_iter', nargs='?', type=int, default = 300,
                            help="Epoch num, Default 300")

        parser.add_argument('-batch', nargs='?', type=int, default=128,
                            help="Batch size, Default 128")

        parser.add_argument('-loss', nargs='?', type=str, choices=list(loss_dic.keys()),default=list(loss_dic.keys())[0],
                            help="Loss function choose:{}, Default:{}".format(list(loss_dic.keys()),list(loss_dic.keys())[0]))

        parser.add_argument('-opt', nargs='?', type=str, choices=['adam','sgd'], default = 'adam',
                            help="Optimizer type: ['adam','sgd'],   Default 'adam'")

        parser.add_argument('-lr', nargs='?', type=float, default=0.001,
                            help="Learning rate, Default 0.001 (1e-3)")

        parser.add_argument('-weight_decay', nargs='?', type=float, default=0.0001,
                            help="Weight_decay, Default 0.0001 (1e-4)")

        parser.add_argument('-lr_step', nargs='*', type=int, default = [30],
                            help="Every 'lr_step' epoch, 'lr' decay, Default 50")

        parser.add_argument('-lr_decay', nargs='?', type=float, default = 0.6,
                            help="Learning rate decay rate, Default 0.6")

        parser.add_argument('-resume', nargs='?', type=str, default = None,
                            help="Resume from the checkpoint_address you give")
    else:
        parser.add_argument('-resume', nargs='+', type=str, default=None,
                            help="Resume from the checkpoint_address you give")
    return parser


class weak_ErrorRate():
    def __init__(self):
        super(weak_ErrorRate,self).__init__()
        self.reset()

    def reset(self):
        self.total = 0
        self.error = 0

    def value(self):
        return [self.error/self.total]

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
        

# ---------------------------------------- Architecture ---------------------------------------------
class FrameWork(object):
    def __init__(self,args):
        super(FrameWork,self).__init__()
        timestamp = datetime.now().strftime('%m%d_%H[%M]')
        self.device = torch.device('cuda', args.gpu) if torch.cuda.is_available() else torch.device('cpu')
        
        if args.resume is not None:
            self.name = findall(r"checkpoints/(.+?).pth",args.resume[0])[0]
            self.resume()
            if self.start >= self.args['max_iter'] - 1 and 'max_iter' in args.__dict__:
                self.args['max_iter'] += args.max_iter
        else:
            self.name = timestamp+'_'+args.n if args.suf=='' else timestamp+'_'+args.n+'_'+args.suf
            if isExist('./checkpoints/'+self.name+'.pth'):
                raise('[{}] has been used!'.format(self.name))

            self.args = args.__dict__
            self.best = 1000
            self.start = 0
            self.build_model()
            with open('./logs/'+self.name+'.log', 'a') as file:
                file.write(str(args.__dict__)+'\n')

        
    def resume(self):
        if isExist('./checkpoints/'+self.name+'.pth'):
            state = torch.load('./checkpoints/'+self.name+'.pth',map_location=lambda storage, loc: storage)

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
        self.net = net_dic[self.args['n']].to(self.device)
        if self.args['opt'] == 'adam':
            self.opt = torch.optim.Adam(self.net.parameters(), lr = self.args['lr'], weight_decay = self.args['weight_decay'])
        elif self.args['opt'] == 'sgd':
            self.opt = torch.optim.SGD(self.net.parameters(), lr = self.args['lr'], weight_decay = self.args['weight_decay'],momentum=0.9)
        if len(self.args['lr_step'])==1:
            milestones = list(np.arange(0,self.args['max_iter'],self.args['lr_step'][0]))
        else:
            milestones = self.args['lr_step']
        self.sch = torch.optim.lr_scheduler.MultiStepLR(self.opt, milestones, gamma=self.args['lr_decay'], last_epoch=-1)
        # self.sch = torch.optim.lr_scheduler.StepLR(self.opt, step_size = self.args['lr_step'], gamma = self.args['lr_decay'])
        self.criterion = loss_dic[self.args['loss']]


    def save(self,epoch,compare):
        if compare<self.best:
            self.best = compare
            torch.save({
                'best': self.best,
                'epoch':epoch,
                'args':self.args,
                'net': self.net.state_dict(),
                'opt' : self.opt.state_dict(),
                'sch' : self.sch.state_dict(),
            }, './checkpoints/'+self.name+'.pth')


    def log_record(self,epoch,print_dic,save=False):
        if epoch is None:
            log = "[Test]:  "
        else:
            log = 'Epoch {}, '.format(epoch)
        for key,value in print_dic.items():
            if value is not None:
                log += '{}: {}, '.format(key,value.value()[0])
        print(log)
        if save:
            with open('./logs/'+self.name+'.log', 'a') as file:
                file.write(log+'\n')
        


    def train(self,train_loader,validation_loader=None,ErrorClass=weak_ErrorRate):
        if not issubclass(ErrorClass,weak_ErrorRate):
            raise("'ErrorClass' is not the sub class of 'weak_ErrorRate'")
        # create record meter
        loss_meter= meter.AverageValueMeter()
        error_meter = ErrorClass()
        loss_meter2 = meter.AverageValueMeter()
        error_meter2 = ErrorClass()
        print_dic={
            'Train_Loss':loss_meter,
            'Train_Error':error_meter if ErrorClass is not weak_ErrorRate else None,
            'Validation_Loss':loss_meter2 if validation_loader is not None else None,
            'Validation_Error': error_meter2 if (validation_loader is not None) and (ErrorClass is not weak_ErrorRate) else None,
        }
        for epoch in range(self.start,self.args['max_iter']):
            # enable the auto-grad engine
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
                error_meter.add(outputs,targets)
            self.sch.step()
            # validation
            if validation_loader is not None:
                # disable the auto-grad engine
                self.net.eval()
                loss_meter2.reset()
                error_meter2.reset()
                for inputs, targets in tqdm(validation_loader):
                    inputs, targets = inputs.to(self.device), targets.to(self.device)

                    outputs = self.net(inputs)
                    loss = self.criterion(outputs, targets)

                    loss_meter2.add(loss.item())
                    error_meter2.add(outputs,targets)
            # save
                if ErrorClass is not weak_ErrorRate:
                    self.save(epoch,error_meter2.value()[0])
                else:
                    self.save(epoch,loss_meter2.value()[0])
            else:
                if ErrorClass is not weak_ErrorRate:
                    self.save(epoch, error_meter.value()[0])
                else:
                    self.save(epoch, loss_meter.value()[0])
            # record
            self.log_record(epoch,print_dic,True)


    def evaluate(self,test_loader,ErrorClass=weak_ErrorRate):
        if not issubclass(ErrorClass,weak_ErrorRate):
            raise("'ErrorClass' is not the sub class of 'weak_ErrorRate'")
        # Cancel auto-grad engine
        self.net.eval() 
        loss_meter = meter.AverageValueMeter()
        error_meter = ErrorClass()
        print_dic={
            'Test_Loss':loss_meter,
            'Test_Error': error_meter if ErrorClass is not weak_ErrorRate else None,
        }
        loss_meter.reset()
        error_meter.reset()
        for inputs, targets in tqdm(test_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            outputs = self.net(inputs)
            loss = self.criterion(outputs, targets)
           
            loss_meter.add(loss.item())
            error_meter.add(outputs,targets)

        self.log_record(None,print_dic)
        


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
        return answer.save(self.name)



if __name__ == '__main__':
    pass
    
