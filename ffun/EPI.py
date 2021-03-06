#!/usr/bin/python
# -*- coding: UTF-8 -*-

from PIL import Image
import os,os.path
import numpy as np
from DataUtil import ImageHelper

class EPI(object):
    '''
    class to create origin epi file
    '''
    def __init__(self, files):
        self.files = files
        self.file_num = len(self.files)
        self.EPIs = []

    def create(self, indexs, axis='u'):
        '''
        function to create epi file\n
        Input:
        - indexs:图像的索引序列
        - axis:'u'--水平方向，'v'--竖直方向
        - folder:to place the epi files generated
        '''
        # check
        assert axis=='u' or axis == 'v'
        assert isinstance(indexs, tuple) or isinstance(indexs, list)
        assert len(indexs) > 0
        #读取图像
        images = []
        for index in indexs:
            f = self.files[index]
            images.append(ImageHelper().read(f))
        # 获取原图像信息
        height, width = images[0].size_H_W()
        channels = images[0].channels()
        # 构建shape
        shape = (len(indexs), width, channels)
        line_num = height
        if axis == 'v':
            shape = (height, len(indexs), channels)
            line_num = width
        # 都会舍弃上一次产生的EPI
        self.EPIs = []#为保证create()对于同一个对象可复用
        # 构建EPI
        for line_index in xrange(line_num):
            epi = np.ndarray(shape)
            cnt = 0
            for image in images:
                #获得numpy.ndarray类型的data
                data = image.data_up()
                line = None
                if axis == 'u':#水平方向上
                    line = data[line_index, :, :]
                    epi[cnt, :, :] = line
                elif axis == 'v':#竖直方向上
                    line = data[:, line_index, :]
                    epi[:, cnt, :] = line
                cnt += 1#cnt自增
            # 持有EPI
            self.EPIs.append(epi)
        print 'done!'
        return self
    def EPIs_data(self):
        '''
        拿到EPI数据.便于以图片为单位infer\n
        Return:
        numpy.ndarray类型list
        '''
        assert len(self.EPIs) != 0
        return self.EPIs
    def save(self, folder, prefix=None, suffix='.png'):
        '保存EPIs'
        # 检测是否存在目录，不存在则创建一个
        if not os.path.exists(folder):
            os.mkdir(folder)
        folder += '/'
        if prefix is not None:
            folder += prefix + '_'
        index = 0
        for epi in self.EPIs:
            filename = folder + '{:0>3}'.format(index) + suffix
            img = Image.fromarray(np.uint8(epi))
            img.save(filename)
            index += 1#index自增
        return self

class PatchHelper(object):
    def __init__(self, image):
        # 只接受numpy.ndarray类型
        assert isinstance(image, np.ndarray)
        self.__image = ImageHelper(image)
        self.__patches = []
    def padding(self, pad):
        '补0填充，pad是一个四维向量，分别表示上下左右的pad数量'
        # check
        assert isinstance(pad, tuple) or isinstance(pad, list)
        assert len(pad) == 4
        image = self.__image
        h, w = image.size_H_W()
        c = image.channels()
        up, down, left, right = pad#得到上下左右的padding数量
        shape = (h + up + down, w + left + right, c)
        data = np.zeros(shape)#定义shape的ndarray，并用0填充
        #批量赋值
        data[up:up + h, left:left + w, :] = image.data_up()
        #持有数据
        self.__image = ImageHelper(data)
    def extract(self, ksize, stride):
        '''
        提取函数：ksize--卷积核尺寸，也就是要提取的patch大小，stride步长\n
        Inputs:
        - ksize:[H,W]
        - stride:[H,W]
        '''
        # check
        assert isinstance(ksize, tuple) or isinstance(ksize, list)
        assert isinstance(stride, tuple) or isinstance(stride, list)
        assert len(ksize) == 2 and len(stride) == 2
        # get image info
        height, width = self.__image.size_H_W()
        kh, kw = ksize
        sh, sw = stride
        start = [0, 0]#初始化卷积左上角坐标
        while start[0] < height:
            #得到卷积左下角H坐标
            end_h = start[0] + kh
            #如果超过图像高度退出循环
            if end_h > height:
                break
            start[1] = 0#初始化卷积左上角W坐标
            while start[1] < width:
                #得到卷积右上角W坐标
                end_w = start[1] + kw
                if end_w <= width:
                    patch = self.__image.data_up()[start[0]:end_h, start[1]:end_w, :]
                    self.__patches.append(patch)
                    start[1] += sw#卷积左上角w坐标右移步长w
                else:
                    break
            start[0] += sh#卷积左上角h坐标下移步长h
        return self
    def patches(self):
        '得到patch'
        assert self.size() > 0
        return self.__patches
    def save(self, folder, prefix='patch', suffix='.png'):
        '保存patch'
        assert self.size() > 0
        # 如果不存在目录,那么创建一个
        if not os.path.exists(folder):
            os.mkdir(folder)
        folder += '/'+prefix
        cnt = 0
        for patch in self.__patches:
            img = Image.fromarray(np.uint8(patch))
            filename = folder + '_{:0>6}'.format(cnt) + suffix
            img.save(filename)
    def size(self):
        '获取对象持有的patch数量'
        return len(self.__patches)

