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


class EPIextractor(object):
    '''
    class to extract EPI window from origin EPI File
    '''
    def __init__(self, epi_file):
        self.file = epi_file
        self.pad_mode = None
        self.pad_num = 0
        self.im_array = None
    def set_padding(self, pad_num, mode='c=0'):
        '''
        function to set padding\n
        @pad_num:图像单侧需要pad的数量,对epi而言,只是在图片两侧补pad,即增加图片的width。
        单侧增加的width为提取时的length的1/2\n
        @mode:\n
        1)constant mode\n
        'c=0'-->constant and value is 0,'c=1'-->constant and value is 1
        2)mirror mode\n
        '''
        self.pad_num = pad_num
        self.pad_mode = mode
        #load the origin epi file
        if self.im_array is None:
            self.__loadfile()
        shape = self.im_array.shape
        width = shape[1]#获得宽度
        v = 0
        if mode[0] == 'c':#constant mode
            v = int(mode[2])
        elif mode[0] == 'm':#mirror mode,not realize
            pass
        nd = np.zeros([shape[0], shape[1]+pad_num*2, shape[2]])
        nd[:, pad_num:shape[1]+pad_num:, :] = self.im_array# copy the data into padded area
        self.im_array = nd# replace the im_marry

    def __loadfile(self):
        '''
        function to load origin epi file and store it with numpy ndarray
        '''
        with open(self.file, 'r') as f:
            im = Image.open(f)
            self.im_array = np.array(im)#转换图像为numpy形式，深拷贝
    def extract(self, point_x, length=33):
        '''
        function to extract a window of origin EPI file\n
        @point_x:point's x coordinate\n
        if extract not in padding mode,point_x is in[length/2,width - length/2]\n
        else it's in [0,width-1]\n
        @length:window's Length,length应该是一个奇数
        '''
        if self.im_array is None:#若持有的im_array是None，则获取到它的值。(lazy loading)
            self.__loadfile()
        #get x's coordinate,坐标转换
        x = point_x
        shape = self.im_array.shape
        #如果该epi设置了padding,那么提取的坐标点将发生偏移,需进行坐标转换
        if self.pad_mode != None:#judge if the padding is actived
            assert x >= 0 and x <= shape[1]#x belong [0,img_width]
            x = x + length/2
        else:
            assert x >= length/2 and x <= shape[1]-length/2 - 1
        #获得epi切片。因为ndarray切片时是到后项index前一个的，所以后项index要+1
        extract_epi = self.im_array[:, x-length/2:x+(length)/2+1, :]
        return extract_epi

    def save_extract(self, point_x, folder, length=33):
        '''
        extract the epi patch and save it
        '''
        epi = self.extract(point_x, length)
        epi_pre_fn = os.path.split(self.file)[1]#get the origin filename
        suffix = ".png"
        #generate the new name
        #文件名规则:原文件名+point_x+原文件的后缀名,表示该epi是在某文件的某点处提取的
        epi_fn = folder+'/'+epi_pre_fn[:epi_pre_fn.find(suffix)]+'_{:0>3}'.format(point_x)+suffix
        #new image & change the data into uint8
        epi_img = Image.fromarray(np.uint8(epi))
        epi_img.save(epi_fn)

