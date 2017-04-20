#!/usr/bin/python
# -*- coding: UTF-8 -*-

'''
author:fang.junpeng
time:2017-04-20
生成EPI数据集
本文件在执行时，需要放到LFDL目录下
'''
from ffun.EPI import EPI
from ffun.FileHelper import FileHelper
import os

def row_indexs(size, row_num):
    'size:矩阵的尺寸，row_num：行号，从0开始'
    height, width = size
    assert 0 <= row_num < height
    row_start = row_num * width
    indexs = []
    for i in xrange(width):
        indexs.append(row_start + i)
    return indexs

def col_indexs(size, col_num):
    'size:矩阵的尺寸，col_num：列号，从0开始'
    height, width = size
    assert 0 <= col_num < width
    indexs = [col_num]
    for i in xrange(1, height):
        indexs.append(col_num + i * width)
    return indexs

def create_epi(src_dir, dst_dir):
    'src_dir:图片路径，dst_dir:目标路径'
    #check
    assert os.path.exists(src_dir) and os.path.isdir(src_dir)
    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)
    files = FileHelper.get_files(src_dir, 'input')
    assert len(files) == 81
    epi = EPI(files)
    size = (9, 9)
    epi.create(row_indexs(size, 4), 'u').save(dst_dir + '/EPI-u')
    epi.create(col_indexs(size, 4), 'v').save(dst_dir + '/EPI-v')

def batch_create(src_dir, dst_dir):
    '对src_dir下的所有文件夹内的数据，都进行EPI生成'
    folders = FileHelper.get_folders(src_dir)
    for folder in folders:
        epi_path = dst_dir + folder[folder.rfind('/'):]
        create_epi(folder, epi_path)

# 1th
#train_data='/home/cs505/workspace/LF_Data/full_data/stratified'
#EPI_dir = '/home/cs505/workspace/LF_Data/EPI/stratified'
# 2th
#train_data='/home/cs505/workspace/LF_Data/full_data/additional'
#EPI_dir = '/home/cs505/workspace/LF_Data/EPI/additional'

# 3th
train_data='/home/cs505/workspace/LF_Data/full_data/training'
EPI_dir = '/home/cs505/workspace/LF_Data/EPI/training'

if __name__ == '__main__':
    batch_create(train_data, EPI_dir)



