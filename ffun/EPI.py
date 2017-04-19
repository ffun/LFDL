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

    def create(self, indexs, axis='u', folder=None):
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
        #get the folder if param is None
        epi_folder = folder
        if epi_folder is None:
            filename = self.files[0]
            epi_folder = os.path.split(filename)[0]
            epi_folder += '/EPI_' + axis + '_' + str(indexs[0])+'_'+str(indexs[-1])
        # 检测是否存在目录，不存在则创建一个
        if not os.path.exists(epi_folder):
            os.mkdir(epi_folder)
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
        # png文件后缀
        png_suffix = '.png'
        # 构建EPI
        for line_index in xrange(line_num):
            epi = np.ndarray(shape)
            cnt = 0
            for image in images:
                #获得numpy.ndarray类型的data
                data = image.data_convert3d()
                line = None
                if axis == 'u':#水平方向上
                    line = data[line_index, :, :]
                    epi[cnt, :, :] = line
                elif axis == 'v':#竖直方向上
                    line = data[:, line_index, :]
                    epi[:, cnt, :] = line
                cnt += 1#cnt自增
            # 保存epi
            filename = '{:0>3}'.format(line_index) + png_suffix
            filename = epi_folder + '/' + filename
            img = Image.fromarray(np.uint8(epi))
            img.save(filename)
        print 'done!'

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
        data[up:up + h, left:left + w, :] = image.data_convert3d()
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
                    patch = self.__image.data_convert3d()[start[0]:end_h, start[1]:end_w, :]
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
    def save(self, folder):
        '保存patch'
        assert self.size() > 0
        cnt = 0
        for patch in self.__patches:
            img = Image.fromarray(np.uint8(patch))
            filename = folder + '/patch_' + '{:0>3}'.format(cnt) + '.png'
            img.save(filename)
    def size(self):
        '获取对象持有的patch数量'
        return len(self.__patches)

