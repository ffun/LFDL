#!/usr/bin/python
# -*- coding: UTF-8 -*-
import os,os.path

class FileHelper(object):
    '''
    class to help deal with folder and File
    '''
    @staticmethod
    def get_files(folder, match_str):
        '''
        获取当前目录下的所有文件\n
        Inputs:
        - folder:目录
        - match_str:需要匹配的文件吗字符串
        - return: tuple
        '''
        files = []
        filelist = os.listdir(folder)
        for f in filelist:
            if os.path.isfile(folder+'/'+f):
                filename = folder+'/'+f#get filename
                if filename.find(match_str) != -1:#filter the file
                    files.append(filename)
        files.sort()#对list的内容进行排序
        return tuple(files)#返回tuple类型
    @staticmethod
    def get_folders(folder):
        '获取当前目录下的所有文件夹'
        folders = []
        if not os.path.isdir(folder):
            return folders
        for s in os.listdir(folder):
            newDir = os.path.join(folder, s)
            if os.path.isdir(newDir):
                folders.append(newDir)
        if len(folders) == 0:
            #说明folader是一个目录，且目录下没有其他目录，那就添加自身吧
            folders.append(folder)
        return folders

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
