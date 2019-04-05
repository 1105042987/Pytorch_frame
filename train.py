#!/usr/bin/env
# -*- coding: utf-8 -*-
''' 
Copyright (c) 2019 Sufer_Qin
'''

# args
import argparse
# time
import time
# ---------------------------------- Network Architecture ----------------------------------
# Define architecture of the network
from net.LeNet          import LeNet
from net.DenseNet       import DenseNet
from net.googlenet      import GoogLeNet
from net.mobilenetv2    import MobileNetV2
from net.vgg            import VGG
from net.wrn            import WideResNet

net_dic = {
    'le':LeNet(), 
    'goo':GoogLeNet(),
    'den':DenseNet(), 
    'mob':MobileNetV2(10), 
    'vgg':VGG('VGG19'),
    'wrn':0,
}

def refresh_net_dic(net,arg):
    net['wrn'] = WideResNet(arg.wrn[0], 10, widen_factor=arg.wrn[1], dropRate=arg.wrn[2])

# user code
from dataset import *
from body import *

def get_args():
    parser = base_args()
    parser.add_argument('--ans', action='store_true',
                        help='Do not use validation data to calc error')
    parser.add_argument('--wrn', nargs=3, type=float, default = [28,1,0.2],
                        help='WideResNet param: [depth,widen_factor,dropRate]')

    return parser.parse_args()

class ErrorRate(weak_ErrorRate):
    def __init__(self):
        super(ErrorRate,self).__init__()

    def add(self,out,tar):
        import torch
        self.total += len(tar)
        outputs = torch.max(out, 1)[1]
        self.error += torch.sum(outputs != tar).float()

class AnsGet(weak_GetPredictAnswer):
    def __init__(self):
        super(AnsGet,self).__init__()

    def add(self,outputs,index):
        from torch import cat
        if self.data is None:
            self.data = cat((index, onehot2num(outputs).to('cpu')), 1)
        else:
            outputs = cat((index, onehot2num(outputs).to('cpu')), 1)
            self.data = cat((self.data,outputs))

    def save(self,name):
        import pandas as pd
        ans = pd.DataFrame(np.array(self.data),columns = ['ID','Category'])
        ans.to_csv('.\\output\\{}.csv'.format(name),header=True,index = False)


if __name__ == "__main__":
    args = get_args()
    print(args)
    
    BODY = Body(args)

    if args.ans:
        trainloader = loadData("TRAIN", None, args.batch)
        testloader = loadData("TEST",None,args.batch)
        BODY.train(trainloader,None,ErrorRate)
        BODY.predict(testloader,AnsGet)
    else:
        trainloader = loadData("train", 1, args.batch)
        validloader = loadData("validation", 1, args.batch)
        BODY.train(trainloader,validloader,ErrorRate)
      