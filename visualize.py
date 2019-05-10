#!/usr/bin/env
# -*- coding: utf-8 -*-
''' 
Copyright (c) 2019 Sufer_Qin
'''

from re import findall
from frame.frame import base_args
import argparse
import utils.draw as draw

def get_args():
    parser = base_args(omit=True)
    parser.add_argument('-skip',  nargs='?', type=int, default=0,
                        help='input how many epoch you want to skip')
    parser.add_argument('-font',  nargs='?', type=int, default=12,
                        help='input the front size you want')

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    name = findall(r"checkpoints/(.+?).pth", args.resume[0])[0]
    draw.drawLossLine(name,args.skip,args.font)
    if args.skip !=0:
        draw.drawLossLine(name, args.skip, args.font,True)
