#!/usr/bin/python
# -*- coding: UTF-8 -*-
'''
file:batch Helper
@author:fang.junpeng
@email:tfzsll@126.com
'''

import random
from Checker import Checker

class BatchHelper(object):
    '''
    Class for shuffle the data-batch\n
    内部实现主要是对索引乱序，并实现对游标计数\n
    '''
    def __init__(self, m_items):
        '''
        构造函数入参为多个数据序列，这些序列的长度要一致
        '''
        self.m_items = m_items
        self.index = [i for i in range(len(self.m_items[0]))]
        self.front = 0
        self.end = len(m_items)
        self.__check()#校验，确保正确
    def __check(self):
        '''
        数据校验
        '''
        #类型校验
        Checker.type_check(self.m_items, (list, tuple))
        #长度校验
        seq_length = len(self.m_items[0])
        for i in range(len(self.m_items)):
            Checker.seq_len_check(self.m_items[i], seq_length)

    def shuffle(self):
        '''
        shuffle the m_items
        '''
        random.shuffle(self.index)
    def reset_cursor(self):
        '''
        make the cursor be initialize
        '''
        self.front = 0
    def re_shuffle(self):
        '''
        re shuffle the m_items
        '''
        self.reset_cursor()
        self.shuffle()
    def head(self):
        '''
        function to get the head item\n
        返回所有序列的队头元素组成的元组
        '''
        item = None
        if self.end - self.front != 1:
            item = []
            for i in range(len(self.m_items)):
                #get the head of the all the item-queue
                item.append(self.m_items[i][self.index[self.front]])
                self.front = self.front + 1
        return tuple(item)

