#!/usr/bin/python
# -*- coding: UTF-8 -*-
'''
数据预处理时的工具
'''
import numpy as np
from PIL import Image

class ImageHelper(object):
    '图像助手'
    def __init__(self, data=None):
        if data is not None:
            # 只接受np.ndarray类型
            assert isinstance(data, np.ndarray)
            # 只接受图像
            assert len(data.shape) > 2
        self.__data = data
        self.__channels = 0
    def read(self, path):
        'read an image,return numpy ndarray'
        with open(path, 'r') as f:
            img = Image.open(f)
            self.__data = np.array(img)
        return self
    def channels(self):
        'get num of channels of the image'
        #计算channels
        shape = self.shape()
        channel = 1
        if len(shape) == 3:
            channel = shape[-1]
        self.__channels = channel#对象持有channels
        return channel
    def data(self):
        'get data'
        assert self.__data is not None
        return self.__data
    def data_up(self):
        '如果图像是灰度图，那么转换成三维形式并返回'
        #灰度图的shape是类似(512,512)的二维形式，需要转化成(512,512,1)的三维形式
        if self.channels() == 1:
            shape = list(self.__data.shape)
            shape.append(1)
            data = np.ndarray(shape)
            h, w = self.size_H_W()
            for i in xrange(h):
                for j in xrange(w):
                    data[i, j, 0] = self.__data[i, j]
            self.__data = data
        return self.__data
    def save(self, path):
        '保存图像'
        # check，防止出现灰度图，而且被up后的数据保存，否则会出错
        if len(self.shape()) == 3:
            assert self.channels() == 3
        img = Image.fromarray(np.uint8(self.data()))
        img.save(path)
        return self
    def shape(self):
        '得到shape'
        return self.data().shape
    def size_H_W(self):
        '得到尺寸:H,W'
        return self.shape()[:2]

class ImageCollection(object):
    'image集合，集合内的元素channel可以不一样，但是size要一样'
    def __init__(self, *images):
        self.IMAGE = []
        self.CHANNEL = 0
        self.SIZE_H_W = ()
        for image in images:
            self.add_image(image)
    def add_image(self, image):
        # 只接受ImageHelper类型
        assert isinstance(image, ImageHelper)
        # 确保加入的image的size和以前的之前持有的image都一样
        if self.size() == 0:
            self.SIZE_H_W = image.size_H_W()
        else:
            assert self.SIZE_H_W == image.size_H_W()
        # 添加image
        self.IMAGE.append(image)
        self.CHANNEL += image.channels()
        return self
    def contact_with_channel(self):
        '所持有的image通过channel叠成一个ndarray'
        assert self.size() != 0
        #生成需要的shape
        shape = list(self.SIZE_H_W)
        shape.append(self.CHANNEL)
        #根据shape生成需要的ndarray
        data = np.ndarray(shape)
        #循环赋值,得到叠在一起的ndarray
        start_ch, end_ch = 0, 0#初始化开始通道和结束通道
        for image in self.IMAGE:
            #更新开始通道和结束通道
            start_ch, end_ch = end_ch, end_ch + image.channels()
            #批量赋值
            data[:, :, start_ch:end_ch] = image.data_up()[:, :, :]
        return data
    def size(self):
        '集合持有的对象数量'
        return len(self.IMAGE)

