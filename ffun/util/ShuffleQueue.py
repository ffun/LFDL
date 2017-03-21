#!/usr/bin/python
# -*- coding: UTF-8 -*-
'''
file:Queue.py
@author:fang.junpeng
@email:tfzsll@126.com
'''

import random

class ShuffleQueue(object):
    '''
    Class for shuffle the data-batch\n
    内部实现主要是对索引乱序，并实现对游标计数
    '''
    def __init__(self, items):
        self.items = items
        self.index = [i for i in range(len(self.items))]
        self.front = 0
        self.end = len(items)
    def shuffle(self):
        '''
        shuffle the items
        '''
        random.shuffle(self.index)
    def reset_cursor(self):
        '''
        make the cursor be initialize
        '''
        self.front = 0
    def re_shuffle(self):
        '''
        re shuffle the items
        '''
        self.reset_cursor()
        self.shuffle()
    def get_item(self):
        '''
        function to get the head item
        '''
        item = None
        if self.end - self.front != 1:
            item = self.items[self.index[self.front]]#get the head of the queue
            self.front = self.front + 1
        return item
