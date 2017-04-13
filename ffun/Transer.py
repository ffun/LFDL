#!/usr/bin/python
# -*- coding: UTF-8 -*-
import numpy as np
from PIL import Image

def image2ndarray(filepath):
    with open(filepath,'r') as f:
        im = Image.open(f)
        return np.array(im)

def im_con_Channel(nds):
    '''
    将多个ndarray表示的图像通过channel合为一个
    '''
    item = nds[0]
    shape = item.shape
    assert len(shape) == 3#验证shape是三维
    channels = shape[-1]
    new_shape = []
    #generate new shape
    for i in xrange(len(shape)):
        if i == len(shape) - 1:
            new_shape.append(shape[i]*len(nds))
        else:
            new_shape.append(shape[i])
    #generate ndarray object
    comb = np.ndarray(new_shape)
    #generate the combined ndarray
    for i in xrange(len(nds)):
        comb[:, :, channels*i:channels*(i+1)] = nds[i][:, :, :]
    return comb
