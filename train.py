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
import dataset.dataset as D
# Define architecture of the network
from frame.net.LeNet          import LeNet
from frame.net.DenseNet       import DenseNet121
from frame.net.googlenet      import GoogLeNet
from frame.net.mobilenetv2    import MobileNetV2
from frame.net.vgg            import VGG
from frame.net.wrn            import WideResNet

net_dic = {
    'le':LeNet(), 
    'goo':GoogLeNet(),
    'den':DenseNet121(), 
    'mob':MobileNetV2(10), 
    'vgg':VGG('VGG19'),
    'wrn':0,
}

def refresh_net_dic(net,arg):
    net['wrn'] = WideResNet(int(arg['wrn'][0]), 10, widen_factor=int(arg['wrn'][1]), dropRate=arg['wrn'][2])

# frame code
import frame.frame as FR

def get_args():
    parser = FR.base_args()
    parser.add_argument('-ans', action='store_true',
                        help='Choose this if you want to use all the train_data to train your model')
    parser.add_argument('-nval', action='store_true',
                        help='Choose this if you do not want to use validation_data')
    parser.add_argument('-wrn', nargs=3, type=float, default = [28,1,0.2],
                        help='WideResNet param: [depth,widen_factor,dropRate]')

    return parser.parse_args()



def train():
    args = get_args()
    print(args)

    BODY = FR.FrameWork(args)

    if 'ans' in BODY.args and BODY.args['ans']:
        trainloader = D.loadData("TRAIN", None, BODY.args['batch'])
        testloader = D.loadData("TEST", None, BODY.args['batch'])
        BODY.train(trainloader, None)
        BODY.predict(testloader, D.AnsGet)
    else:
        trainloader = D.loadData("train", None, BODY.args['batch'])
        if 'nval' in BODY.args and not BODY.args['nval']:
            validloader = D.loadData("validation", None, BODY.args['batch'])
        else:
            validloader = None
        BODY.train(trainloader, validloader, D.ErrorRate)


if __name__ == "__main__":
    train()
      
