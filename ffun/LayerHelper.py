#!/usr/bin/python
# -*- coding: UTF-8 -*-
'''
author:fang.junpeng\n
email:tfzsll@126.com\n
'''
import math
import tensorflow as tf
import NetHelper

class LayerHelper(object):
    '''
    class for easy build net
    '''
    @classmethod
    def weight(cls, shape, stddev=1e-2, name=None):
        '''
        funciton to get weight-param of tf.Variable type
        '''
        initial = tf.truncated_normal(shape, stddev=stddev)
        return tf.Variable(initial, name=name)
    @classmethod
    def bias(cls, shape, value=1e-2, name=None):
        '''
        funciton to get bias-param of tf.Variable type
        '''
        initial = tf.constant(value=value, shape=shape)
        return tf.Variable(initial, name=name)
    @classmethod
    def conv(cls, x, w, strides=[1, 1, 1, 1], padding='VALID', name=None):
        '''
        function to do convolution compute\n
        @x:input tensor\n
        @w:weight of the convolutional kernel\n
        @strides:the step of conv kernel walk on every dimension\n
        @padding:padding mode,'VALID'--without padding,"SAME" = with zero padding
        '''
        return tf.nn.conv2d(x, w, strides=strides, padding=padding, name=name)
    @classmethod
    def pool(cls, x, ksize, strides, padding='VALID', style="max", name=None):
        '''
        function to do pool compute\n
        @x:input tensor\n
        @ksize:size of the  kernel\n
        @strides:the step of conv kernel walk on every dimension\n
        @padding:padding mode,'VALID'--without padding,"SAME" = with zero padding
        @style:max,or avg
        '''
        if style == "max":
            return tf.nn.max_pool(x, ksize=ksize, strides=strides, padding=padding, name=name)
        elif style == 'avg':
            return tf.nn.avg_pool(x, ksize, strides=strides, padding=padding, name=name)

class TfBuilder(NetHelper.NetBuilder):
    'Tensorflow Net builder'
    def conv2d(self, x, w_shape, strides, padding, name, initializer_w=None, initializer_b=None):
        '''
        convolution layer:
        Input
        - x:input tensor
        - w_shape:weight shape for convolution kernel
        - strides
        - padding:'SAME' or 'VALID'
        - name:variable name scope
        - initializer_w/b:initializer of weight and bias
        '''
        _, _, _, num_out = w_shape
        with tf.variable_scope(name) as scope:
            weights = tf.get_variable('weights', w_shape, initializer=initializer_w)
            biases = tf.get_variable('biases', [num_out], initializer=initializer_b)
        #conv
        conv = tf.nn.conv2d(x, weights, strides, padding)
        #relu
        relu = tf.nn.relu(conv + biases, name=scope.name)
        return relu
    def max_pool(self, x, ksize, strides, padding, name=None):
        'max pooling layer'
        return tf.nn.max_pool(x, ksize, strides, padding)
    def fc(self, x, w_shape, name, relu=True, initializer_w=None, initializer_b=None):
        'fully connected layer'
        _, num_out = w_shape
        with tf.variable_scope(name) as scope:
            weights = tf.get_variable('weights', w_shape, initializer=initializer_w)
            biases = tf.get_variable('biases', [num_out], initializer=initializer_b)
        act = tf.nn.xw_plus_b(x, weights, biases, name=scope.name)
        if relu:
            relu = tf.nn.relu(act)
            return relu
        return act
    def dropout(self, x, keep_prop):
        'keep_prpo layer'
        return tf.nn.dropout(x, keep_prop)
