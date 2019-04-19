#!/usr/bin/env
# -*- coding: utf-8 -*-
''' 
Copyright (c) 2019 Sufer_Qin
'''
from frame.frame import base_args,FrameWork
# args
import argparse
# time
import time

def get_args():
    parser = base_args(omit = True)
    parser.add_argument('-ans', action='store_true',
                        help='Choose this if you want to get the answer of test_data')
    return parser.parse_args()

from train import D
def evaluate():
    args = get_args()
    print(args)

    BODY = FrameWork(args)

    if args.ans:
        testloader = D.loadData("TEST", None, BODY.args['batch'])
        BODY.predict(testloader, D.AnsGet)
    else:
        validloader = D.loadData("validation", None, BODY.args['batch'])
        BODY.evaluate(validloader, D.ErrorRate)

if __name__ == "__main__":
    evaluate()
