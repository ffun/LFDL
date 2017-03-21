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
        function:to get all files in the current folder\n
        return: tuple\n
        @suffix:file's suffix
        '''
        files = []
        filelist = os.listdir(folder)
        for f in filelist:
            if os.path.isfile(folder+'/'+f):
                filename = folder+'/'+f#get filename
                if filename.find(suffix) != -1:#filter the file
                    files.append(filename)
        files.sort()#对list的内容进行排序
        return tuple(files)#返回tuple类型

class EPIcreator(object):
    '''
    class to create origin epi file
    '''
    def __init__(self, files):
        self.files = files
        self.file_num = len(self.files)

    def create(self, block, folder=None):
        '''
        function to create origin epi file\n
        @block:图像索引闭区间\n
        @folder:to place the epi files generated\n
        default is the folder of origin images in files
        '''
        if not isinstance(block, tuple):
            raise TypeError("block should be tuple-type")
        if len(block) != 2 and block[0]<block[1]:
            raise ValueError("len(block) should be 2")
        start, end = block[0], block[1]
        #get the folder if param is None
        if folder is None:
            filename = self.files[0]
            #folder = filename[:filename.rfind('/')]
            folder = os.path.split(filename)[0]#get the folder
        epi_folder = folder+'/epi'+str(start)+'_'+str(end)+'/'
        #judge the folder exist
        if not os.path.exists(epi_folder):
            os.mkdir(epi_folder)# creat a folder for EPI
        #some prefix & suffix of file
        epi_file_prefix = 'epi_'+str(start)+'_'+str(end)+'_'
        png_suffix = '.png'
        images = self.files
        #get width and height of the image
        width, height = 0, 0
        with open(images[start], 'r') as f:
            im = Image.open(f)
            width, height = im.size
        #generate origin epi
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
            print 'epi'+'{:0>3}'.format(h)+' generate'
        print 'done!'


class EPIextractor(object):
    '''
    class to extract EPI window from origin EPI File
    '''
    def __init__(self, epi_file):
        self.file = epi_file
        self.padding = 0
        self.pad_mode = None
        self.im_array = None
    def set_padding(self, num, mode):
        '''
        function to set padding\n
        @num:number of pixels to be padded\n
        @mode:how to pad the image\n
        '''
        self.padding = num
        self.pad_mode = mode
    def extract(self, point_x, length=31):
        '''
        function to extract a window of origin EPI file\n
        @point_x:point's x coordinate\n
        @length:window's Length,length应该是一个奇数
        '''
        if self.im_array is None:#若持有的im_array是None，则获取到它的值。(lazy loading)
            with open(self.file, 'r') as f:
                im = Image.open(f)
                im_array = np.array(im)#转换图像为numpy形式，深拷贝
        extract_epi = im_array[:, point_x-length/2:point_x+(length)/2, :]
        return extract_epi
        #print extract_epi.shape