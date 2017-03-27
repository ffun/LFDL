#!/usr/bin/python
# -*- coding: UTF-8 -*-
import numpy as np
import Color_pt
from Checker import Checker

class NetCalculator(object):
    '''
    class for Net tool\n
    '''
    def __init__(self):
        self.layers = []
        self.layer_index= 0
        self.shapes = []
        self.shape_index = 0
        self.has_dataLayer = False
    def __append_layer(self, layer):
        '''
        A private funtion to append a layer to Object's layer set
        '''
        self.layers.append(layer)
        self.layer_index = self.layer_index + 1
    
    def __append_shape(self, shape):
        self.shapes.append(shape)
        self.shape_index = self.shape_index + 1
    def __build(self, name, ksize, strides):
        '''
        A private funtion to build layer description
        '''
        layer = {}
        layer['name'] = name
        layer['ksize'] = ksize
        layer['strides'] = strides
        return layer
    def set_dataLayer(self, shape):
        '''
        funtion to set data layer's description\n
        @shape:(H,W,C)
        '''
        layer = self.__build('data-layer', 0, 0)
        shape = np.array(shape, np.int32)# change into numpy
        self.__append_layer(layer)#add layer
        self.__append_shape(shape)#add shape
        self.has_dataLayer = True
    def append_layer(self, layer_type, ksize, strides=[1,1,1,1], padding = 0):
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
        old_shape = self.shapes[-1]
        in_channel_old = old_shape[2]
        Hi = old_shape[0]
        Wi = old_shape[1]
        #shape with (H,W,C)
        new_shape = np.zeros([3], np.int32)
        #check in_channel
        if layer_type == 'conv':
            assert in_channel == in_channel_old
            new_shape[2] = out_channel#assign out_channel to in_channel of shape
        elif layer_type == 'pool':
            new_shape[2] = old_shape[2]
        #compute output shape,(H,W,C)
        H0 = (Hi + 2*padding - Kh)/StrideKh + 1
        W0 = (Wi + 2*padding - Kw)/StrideKw + 1
        new_shape[0] = H0
        new_shape[1] = W0
        #add layer
        name = layer_type + str(self.layer_index + 1)
        layer = self.__build(name, ksize, strides)
        self.__append_layer(layer)
        #add shape
        self.__append_shape(new_shape)
    def num_of_layers(self):
        '''
        get number of layers the Object Holds(include data-layer)
        '''
        return self.layer_index
    def print_layers(self, color='green'):
        '''
        print info && output-shape of every layer
        '''
        for i in xrange(len(self.layers)):
            layer = self.layers[i]
            layer_name = layer['name']
            layer_ksize = layer['ksize']
            layer_strides = layer['strides']
            layer_info = str(layer_name)+'->ksize:'+str(layer_ksize) +\
            ';strides:'+str(layer_strides)
            print layer_info
            shape_info = 'output:' + str(self.shapes[i])
            Color_pt.pt(shape_info, color=color)
