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
    'Class for helping to load file and get data'
    def __init__(self, separator=" "):
        '''
        Inputs:
        - separator:一行中元素的分隔符，默认空格
        '''
        self.separator = separator

    def read(self, path, transform=None):
        '''
        读取标签文件，对每个元素进行转换，返回一个list\n
        Inputs:
        - path:标签文件的路径
        - transform:转换函数
        '''
        with open(path, "rb") as f:
            d = []
            for line in f.readlines():
                line = line.strip()#过滤掉换行符
                data = line.split(self.separator)
                if transform is not None:
                    data = map(transform, data)
                d.extend(data)
        return d
