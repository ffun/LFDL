#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np

class TextLoader(object):
    '''
    Class to help loader file and get data
    '''
    def __init__(self, separator=" "):
        '''
        @separator:the separator of element in one line,\n
        Default is space
        '''
        self.data = []
        self.separator = separator

    def read(self, filepath, transform=float):
        '''
        Function to get a list of data\n
        @filepath:the path of the file\n
        @transform:transform the single data,default:float\n
        Example\n
        1:line in file "1 2\\r\\n"\n
        result:[1,2]\n
        2:line in file "1 2\\r\\n3 4\\r\\n"\n
        result:[[1,2],[3,4]]
        '''
        with open(filepath, "rb") as f:
            for line in f.readlines():
                line = line.strip()#过滤掉换行符
                data = line.split(self.separator)
                data = map(transform, data)
                self.data.append(data)
        return tuple(self.data)

    def getdata(self):
        if len(self.data) != 0:
            return self.data
        else:
            return None
