#!/usr/bin/python
# -*- coding: UTF-8 -*-

'''
author:fang.junpeng\n
email:tfzsll@126.com
time:2017-03-27
'''

import numpy as np
import Color_pt
from Checker import Checker

class NetCalculator(object):
    '''
    class for Net tool\n
    自动判断参数是否正确，以及计算出每一层的输出shape\n
    目前支持conv,pool,fc型处理层
    目前支持线性网络
    '''
    def __init__(self):
        self.layers = []
        self.shapes = []
        self.has_dataLayer = False
        self.layer_stat={'conv':0, 'pool':0, 'fc':0}
    def __append_layer(self, layer, layer_type):
        '''
        A private funtion to append a layer to Object's layer set
        '''
        self.layers.append(layer)
        if not layer_type is None:
            num = self.layer_stat[layer_type]
            self.layer_stat[layer_type] = num +1#update num
    def __append_shape(self, H, W, C, D=3):
        '''
        function add hidden-data-shape\n
        when conv/pool layer,output Dimension = 3,but when fc,Dimension = 1
        '''
        shape = [H, W, C, D]
        self.shapes.append(shape)
    def __build(self, name, ksize, strides, shape=None):
        '''
        A private funtion to build layer description
        '''
        layer = {}
        layer['name'] = name
        layer['ksize'] = ksize
        layer['strides'] = strides
        layer['shape'] = shape
        return layer
    def last_layer(self):
        if not self.has_dataLayer:
            return None
        return self.layers[-1]
    def last_shape(self):
        '''
        function to get last layer shape\n
        return:(H,W,C)
        '''
        if not self.has_dataLayer:
            return 0, 0, 0
        shape = self.shapes[-1]
        return shape[0], shape[1], shape[2]
    def set_dataLayer(self, shape):
        '''
        funtion to set data layer's description\n
        @shape:(H,W,C)
        '''
        layer = self.__build('data', 0, 0)
        self.__append_layer(layer, None)#add layer
        self.__append_shape(shape[0], shape[1], shape[2])#add shape
        self.has_dataLayer = True
    def add_conv_layer(self, ksize, strides=[1, 1, 1, 1], padding=0):
        '''
        funtion to append layer's description\n
        @layer_type:Now,only can be 'conv' or 'pool'\n
        @ksize:(H,W,in_channels,out_channels)\n
        @strides:(N,H,W,C),most case,N = 1 && C = 1\n
        @padding:in here,pad is to number of points all arround the tensor plane
        '''
        #check if has data layer
        if not self.has_dataLayer:
            raise ValueError('The Object lose DataLayer')
        #check length
        Checker.seq_len_check(ksize, 4)
        Checker.seq_len_check(strides, 4)
        #get param
        Kh = ksize[0]
        Kw = ksize[1]
        in_channel = ksize[2]
        out_channel = ksize[3]
        StrideKh = strides[1]
        StrideKw = strides[2]
        #获取末尾shape
        Hi, Wi, Ci = self.last_shape()
        #check in_channel
        assert in_channel == Ci
        #compute output shape,(H,W,C)
        H0 = (Hi + 2*padding - Kh)/StrideKh + 1
        W0 = (Wi + 2*padding - Kw)/StrideKw + 1
        #add layer and shape
        name = 'conv'+ str(self.layer_stat['conv']+1)
        layer = self.__build(name, ksize, strides)
        self.__append_layer(layer, 'conv')
        self.__append_shape(H0, W0, out_channel)
    def add_pool_layer(self, ksize, strides=[1, 1, 1, 1], padding=0):
        '''
        function to add pool layer
        '''
        #check if has data layer
        if not self.has_dataLayer:
            raise ValueError('The Object lose DataLayer')
        #check length
        Checker.seq_len_check(ksize, 4)
        Checker.seq_len_check(strides, 4)
        #get param
        Kh = ksize[1]
        Kw = ksize[2]
        StrideKh = strides[1]
        StrideKw = strides[2]
        #获取末尾shape
        Hi, Wi, Ci = self.last_shape()
        #compute output shape,(H,W,C)
        H0 = (Hi + 2*padding - Kh)/StrideKh + 1
        W0 = (Wi + 2*padding - Kw)/StrideKw + 1
        #add layer and shape
        name = 'pool'+ str(self.layer_stat['pool']+1)
        layer = self.__build(name, ksize, strides)
        self.__append_layer(layer, 'pool')
        self.__append_shape(H0, W0, Ci)
    def add_fc_layer(self, shape):
        '''
        add fc layer
        '''
        Checker.seq_len_check(shape, 2, "length of shape should be 2")
        #get in-neurons
        in_neurons = shape[0]
        out_neurons = shape[1]
        last_layer = self.last_layer()
        last_shape = self.last_shape()
        old_neurons = 0
        #get old_neurons
        if last_layer['name'].find('fc') != -1:
            old_neurons = last_shape[0]
        else:
            old_neurons = last_shape[0]*last_shape[1]*last_shape[2]
        print old_neurons
        assert old_neurons == in_neurons#check
        #add shape and layer
        name = 'fc'+str(self.layer_stat['fc'] + 1)
        layer = self.__build(name, 0, 0, shape)
        self.__append_layer(layer, 'fc')
        self.__append_shape(out_neurons, 0, 0, D=1)
    def num_of_layers(self):
        '''
        get number of layers the Object Holds(not include data-layer)
        '''
        num = 0
        for k in self.layer_stat:
            num = num + self.layer_stat[k]
        return num
    def print_layers(self, color='green'):
        '''
        print info && output-shape of every layer
        '''
        for i in xrange(len(self.layers)):
            #data layer info
            if i == 0:
                info = 'input:'+str(self.shapes[i])
                Color_pt.pt(info)
                continue
            # other layer info
            layer = self.layers[i]
            shape = self.shapes[i]
            layer_name = layer['name']
            layer_ksize = layer['ksize']
            layer_strides = layer['strides']
            layer_info = str(layer_name)
            shape_info = 'output:'
            if layer_name.find('fc') != -1:
                layer_info = layer_info + '--param:'+str(layer['shape'])
                shape_info = shape_info + str([shape[0]])
            else:
                layer_info = layer_info +'--ksize:'+str(layer_ksize) +';strides:'+str(layer_strides)
                shape_info = shape_info + str([shape[0], shape[1], shape[2]])
            print layer_info
            Color_pt.pt(shape_info, color=color)
    def weight_memery_cost(self, batch_size=1):
        '''
        stat the weight memory cost
        '''
        num = 0
        for i in xrange(len(self.layers)):
            layer = self.layers[i]
            layer_name = layer['name']
            layer_ksize = layer['ksize']
            #data layer
            if layer['name'] == 'data':
                continue
            #conv layer
            if layer['name'].find('conv') != -1:
                #H, W, in, out = layer_ksize[0], layer_ksize[1], layer_ksize[2], layer_ksize[3]
                H = layer_ksize[0]
                W = layer_ksize[1]
                in_channel = layer_ksize[2]
                out_channel = layer_ksize[3]
                # add one bias
                num = num + (H * W * in_channel) * (out_channel + 1)
            if layer['name'].find('fc') != -1:
                shape = layer['shape']
                in_neurons = shape[0]
                out_neurons = shape[1]
                # add one bias
                num = num + in_neurons * (out_neurons + 1)
            #all num
            num = num * batch_size
        return num
    def hidden_memory_cost(self, batch_size=1):
        '''
        stat memory cost of the hidden layer's data
        '''
        num = 0
        for i in xrange(len(self.shapes)):
            shape = self.shapes[i]
            #input-data layer not stat
            if i == 0:
                continue
            D = shape[3]
            if D == 3:
                H, W, C = shape[0], shape[1], shape[2]#get H,W,C
                num = num + H*W*C#add
            elif D == 1:
                num = num + shape[0]
        num = num * batch_size
        return num
    def data_memory_cost(self, batch_size=1):
        '''
        stat memory cost of the input batch data
        '''
        num = 0
        for i in xrange(len(self.shapes)):
            shape = self.shapes[0]
            H, W, C = shape[0], shape[1], shape[2]#get H,W,C
            num = num + H*W*C#add
            break
        num = num * batch_size
        return num
    def all_memort_cost(self, batch_size=1):
        '''
        stat all memory cost of the net when training
        '''
        num = self.weight_memery_cost()+self.hidden_memory_cost()+self.data_memory_cost()
        return num
