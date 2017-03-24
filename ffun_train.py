#!/usr/bin/python
# -*- coding: UTF-8 -*-
import tensorflow as tf
from ffun.util import *

def placeholder_inputs(batch_size, shape):
    '''
    function to generate feed_dict\n
    @batch_size:batch-size of data-batch\n
    @shape:len(shape)=3,for Image Object
    '''
    Checker.seq_len_check(shape, 3)#check the length
    images_placeholder = tf.placeholder(tf.float32, shape=(batch_size, shape[0], shape[1], shape[2]))
    labels_placeholder = tf.placeholder(tf.int32, shape=(batch_size))
    return images_placeholder, labels_placeholder

def fill_feed_dict(data_set, images_pl, labels_pl):
    #batch size 
    current = 
    images_feed, labels_feed = 
    feed_dict = {
        images_pl: images_feed,
        labels_pl: labels_feed,
    }
    return feed_dict