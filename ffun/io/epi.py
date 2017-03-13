#!/usr/bin/python
# -*- coding: UTF-8 -*-

from PIL import Image
import os, os.path
import numpy as np

class FileHelper(object):
    '''
    class to help deal with folder and File
    '''
    @classmethod
    def get_files(cls, folder, suffix='.png'):
        '''
        function:to get all files\n
        return: tuple\n
        @suffix:file's suffix
        '''
        files = []
        for parent, dirnames, filenames in os.walk(folder):
            for filename in filenames:
                if filename.find(suffix) != -1:
                    files.append(os.path.join(parent, filename))
        files.sort()#对list的内容进行排序
        return tuple(files)#返回tuple类型

class EPIcreator(object):
    def __init__(self, folder):
        self.folder = folder
        self.files = FileHelper.get_files(folder)
        self.file_num = len(self.files)

    def create(self, block):
        if not isinstance(block, tuple):
            raise TypeError("block should be tuple-type")
        if len(block) != 2 and block[0]<block[1]:
            raise ValueError("len(block) should be 2")
        start, end = block[0], block[1]
        epi_folder = self.folder+'/epi'+str(block[0])+'_'+str(block[1])+'/'
        os.mkdir(epi_folder)# creat a folder for EPI
        epi_file_prefix = 'epi'+str(block[0])+'_'+str(block[1])+'_'
        png_suffix = '.png'
        images = self.files
        width, height = 0, 0
        with open(images[start], 'r') as f:
            im = Image.open(f)
            width,height = im.size
        for h in range(0, height):
            epi_image = Image.new('RGB', (width, end-start+1), (255, 255, 255))
            for j in range(start, end+1):
                with open(images[j], 'r') as f:
                    im = Image.open(f)
                    for w in range(0, width):
                        pixel = im.getpixel((w, h))#get the pixel
                        epi_image.putpixel((w, j-start), pixel)#put the pixel into epi_image
            #保存epi图像，图像名称后缀,为截取的区间和高度，其中高度使用3位数对齐
            epi_image.save(epi_folder+epi_file_prefix+'{:0>3}'.format(h)+png_suffix)
            print 'epi'+str(h)+' generate'


class EPIextractor:
    def __init__(self, epi_file):
        self.file = epi_file
        self.padding = 0
    def set_padding(self, num):
        self.padding = num
    def extract(self, point, length = 32):
        if not isinstance(point, tuple):
            raise TypeError("point should be tuple-type")
        
