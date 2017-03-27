#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np
import Color_pt
from Checker import Checker

class NetCalculator(object):
    '''
    class for Net tool\n
    自动判断参数是否正确，以及计算出每一层的输出shape\n
    目前仅支持conv,pool型层
    '''
    def __init__(self):
        self.layers = []
        self.shapes = []
        self.shape_index = 0
        self.has_dataLayer = False
        self.layer_stat={'conv':0, 'pool':0}
    def __append_layer(self, layer, layer_type):
        '''
        A private funtion to append a layer to Object's layer set
        '''
        self.layers.append(layer)
        if not layer_type is None:
            num = self.layer_stat[layer_type]
            self.layer_stat[layer_type] = num +1#update num
    def __append_shape(self, H, W, C):
        shape = [H, W, C]
        self.shapes.append(shape)
    def __build(self, name, ksize, strides):
        '''
        A private funtion to build layer description
        '''
        layer = {}
        layer['name'] = name
        layer['ksize'] = ksize
        layer['strides'] = strides
        return layer
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
            layer_name = layer['name']
            layer_ksize = layer['ksize']
            layer_strides = layer['strides']
            layer_info = str(layer_name)+'--ksize:'+str(layer_ksize) +\
            ';strides:'+str(layer_strides)
            print layer_info
            shape_info = 'output:' + str(self.shapes[i])
            Color_pt.pt(shape_info, color=color)
