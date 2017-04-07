#!/usr/bin/python
# -*- coding: UTF-8 -*-
'''
author:fang.junpeng\n
email:tfzsll@126.com\n
'''
import tensorflow as tf

class Layer(object):
    '''
    class for easy build net
    '''
    @classmethod
    def weight_variable(cls, shape, stddev=1e-2, name=None):
        '''
        funciton to get weight-param of tf.Variable type
        '''
        initial = tf.truncated_normal(shape, stddev=stddev, name=name)
        return tf.Variable(initial)
    @classmethod
    def bias_variable(cls, shape, value=1e-2, name=None):
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


