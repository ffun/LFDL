#!/usr/bin/python
# -*- coding: UTF-8 -*-
import os,os.path

class FileHelper(object):
    '''
    class to help deal with folder and File
    '''
    @staticmethod
    def get_files(folder, suffix='.png'):
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

class LabelHelper(object):
    '''
    Class to help loader file and get data
    '''
    def __init__(self, separator=" "):
        '''
        @separator:the separator of element in one line,\n
        Default is space
        '''
        self.separator = separator

    def read(self, filepath, transform=None):
        '''
        Function to get a list of data\n
        @filepath:the path of the file\n
        @transform:transform the single data,default:float\n
        Example:\n
        1:line in file "1 2\\r\\n"\n
        result:[1,2]\n
        2:line in file "1 2\\r\\n3 4\\r\\n"\n
        result:[[1,2],[3,4]]
        '''
        with open(filepath, "rb") as f:
            d = []
            for line in f.readlines():
                line = line.strip()#过滤掉换行符
                data = line.split(self.separator)
                #if transform Function is not None,do transform
                for i in xrange(len(data)):
                    d.append(transform(data[i]))
        return d
