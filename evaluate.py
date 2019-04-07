#!/usr/bin/env
# -*- coding: utf-8 -*-
''' 
Copyright (c) 2019 Sufer_Qin
'''
from train import ErrorRate, AnsGet, loadData
from frame.frame import base_args,FrameWork
# args
import argparse
# time
import time

def get_args():
    parser = base_args(omit = True)
    parser.add_argument('--ans', action='store_true',
                        help='Choose this if you want to get the answer of test_data')
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    print(args)
    
    B = FrameWork(args)
    
    if args.ans:
        testloader = loadData("TEST",None, B.args['batch'])
        B.predict(testloader,AnsGet)
    else:
        validloader = loadData("validation", None, B.args['batch'])
        B.evaluate(validloader,ErrorRate)
