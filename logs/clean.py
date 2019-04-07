#coding=utf-8
# -*- coding: utf-8 -*-
import os
import sys
# sys.setdefaultencoding('utf-8')


def GetFileList(dir, fileList, min_size):
    newDir = dir
    if os.path.isfile(dir):
        fsize = os.path.getsize(dir)/1024
        if fsize < min_size and dir[-2:]!='py':
            fileList[dir[2:-4]] = fsize
    elif os.path.isdir(dir):
        for s in os.listdir(dir):
            newDir = os.path.join(dir, s)
            GetFileList(newDir, fileList, min_size)
    return fileList

def show(dic):
    for key,val in dic.items():
        print(key,':\t',val,' KB')
    print(' ')

if __name__ == "__main__":
    if len(sys.argv) > 1:
        flag = (sys.argv[1]=='cls')
    else: flag = False
    k = input("Clean file under xx KB: ")
    while k.isnumeric():
        L={}
        GetFileList('./', L, int(k))
        show(L)
        k = input("Clean them all?  t/f/new number: ")
        if flag:
            os.system('cls')
    else:
        if k.lower()[0] == 't':
            for item in L.keys():
                os.remove('./'+item+'.log')
                if os.path.exists('../checkpoints/'+item+'.pth'):
                    os.remove('../checkpoints/'+item+'.pth')
        if flag:
            os.system('cls')

