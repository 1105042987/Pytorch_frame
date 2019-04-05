#!/usr/bin/env
# -*- coding: utf-8 -*-
''' 
Copyright (c) 2019 Sufer_Qin
'''
from train import ErrorRate,AnsGet
from dataset import *
from body import *

# args
import argparse
# time
import time

def get_args():
    parser = base_args(list(net_dic.keys()))
    parser.add_argument('--ans', action='store_true',
                        help='Do not use validation data to calc error')

if __name__ == "__main__":
    args = get_args()
    print(args)
    
    B = Body(args)
    
    if args.ans:
        testloader = loadData("TEST",None,args.batch)
        B.predict(testloader,AnsGet)
    else:
        validloader = loadData("validation", None, args.batch)
        B.evaluate(validloader,ErrorRate)
