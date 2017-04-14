#!/usr/bin/python
# -*- coding: UTF-8 -*-
'''
create disp.txt
author:fang.junpeng
'''
import os,os.path
import file_io
import numpy as np

# 1th
#train_data = "/home/cs505/workspace/LF_Data/full_data/training"
#disp_dir='/home/cs505/workspace/LF_Data/disp/train'
# 2th
#train_data = "/home/cs505/workspace/LF_Data/full_data/additional"
#disp_dir = '/home/cs505/workspace/LF_Data/disp/additional'
# 3th
train_data = '/home/cs505/workspace/LF_Data/full_data/stratified'
disp_dir = '/home/cs505/workspace/LF_Data/disp/stratified'

def get_folders(folder):
    folders=[]
    if not os.path.isdir(folder):
        return folders
    for s in os.listdir(folder):
        newDir = os.path.join(folder, s)
        if os.path.isdir(newDir):
            folders.append(newDir)
    if len(folders) == 0:
        #说明folader是一个目录，且目录下没有其他目录，那就添加自身吧
        folders.append(folder)
    return folders

def write_disp(disparity_map, path):
    'record disp into file'
    with open(path, 'w') as f:
        height, weight = disparity_map.shape
        for i in xrange(height):
            dataline = disparity_map[i]
            for j in xrange(weight):
                data = dataline[j]
                f.write(str(data)+' ')
            f.write('\n')

def generate(data_dir, disp_dir):
    '''
    生成disp文件:
    - data_dir:训练数据目录
    - disp_dir:生成的disp文件存放的目录
    '''
    folders = get_folders(data_dir)
    for folder in folders:
        #get disp data
        disparity_map = file_io.read_disparity(folder, highres=False)
        #get disp_path
        disp_path = disp_dir+'/'+folder[folder.rfind('/'):]+'_disp.txt'
        #write disp data
        write_disp(disparity_map, disp_path)

if __name__ == '__main__':
    generate(train_data, disp_dir)
    print 'done!'


