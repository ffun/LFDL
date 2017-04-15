#!/usr/bin/python
# -*- coding: UTF-8 -*-
'''
配置文件
'''
# file params
Label_DIR = '/Users/fang/workspaces/tf_space/LFDL/disp.txt'#标签位置
Image_DIR = '/Users/fang/workspaces/tf_space/box'#图片位置
EPI_DIR = '/Users/fang/workspaces/tf_space/box/epi36_44'#EPI图像位置
Model_DIR = '/Users/fang/workspaces/tf_space/model'#模型位置

#EPI file
EPI_H = 9
EPI_W = 512
EPI_C = 3

# data params
Data_ALL_NUM = 512*512#总数据量
Data_TRAIN_NUM = 200000#训练集数量
Data_VERIFY_NUM = 40000#验证集数量
Data_TEST_NUM = 22000#测试集数量

# input size of EPT Patch
Input_H = 9#图像高度
Input_W = 33#图像宽度
Input_C = 3#图像通道数

# train params
KEEP_PROP = 0.5# dropout率
LR = 1e-4 # 学习率
Batch_SIZE = 50 #batch-size
Iter_SIZE = (Data_TRAIN_NUM//Batch_SIZE)#[batch-size]个训练数据forward+backward后更新参数过程
Epoch_SIZE = 50#一次epoch=所有训练数据forward+backward后更新参数的过程

#TF param
TB_Log_DIR = (Model_DIR+'/tb_log')#TensorBoard
