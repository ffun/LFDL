#!/usr/bin/python
# -*- coding: UTF-8 -*-
'''
file:BatchHelper.py\n
@author:fang.junpeng\n
@email:tfzsll@126.com
'''

import random
from Checker import Checker
import sys

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
        self.__check()#校验，确保正确
        self.index = [i for i in range(len(self.m_items[0]))]
        self.front = 0
        self.end = len(m_items[0])#self.end point to the end of seq
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

    def shuffle(self,times=1):
        '''
        shuffle the m_items
        '''
        self.reset_cursor()
        for i in range(times):
            random.shuffle(self.index)
    def reset_cursor(self):
        '''
        make the cursor be initialize
        '''
        self.front = 0
        self.end = len(self.m_items[0])
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
        if item != None:
            return tuple(item)
        else:
            return item
    def print_data(self):
        '''
        print data according the index\n
        it will not ask for memory for new Object\n
        suggestion:in most cases,this function is used for test
        '''
        index = self.index
        #generate batch
        for i in range(len(self.m_items)):
            item = self.m_items[i]
            for j in range(len(index)):
                elem = item[index[j]]
                sys.stdout.write(str(elem))
            sys.stdout.write(',')
        sys.stdout.write('\n')
    def get_data(self):
        '''
        return data according the index\n
        suggestion:in most cases,this function is used for test
        '''
        data = []
        index = self.index
        #generate batch
        for i in range(len(self.m_items)):
            data_part = []
            item = self.m_items[i]
            for j in range(len(index)):
                elem = item[index[j]]
                data_part.append(elem)
            data.append(data_part)
        return data
    def get_batch(self, batch_size):
        '''
        function to get a batch items of batch-size\n
        it will get batch circularly\n
        suggestion:batch_size should be less than which the Object holds
        '''
        items_num = self.end - self.front#剩余元素的总数
        bz_end = items_num
        index = None
        #set bz circularly && generate index
        if batch_size > items_num and items_num != 0:#当剩余的元素数量比要获取的batch_size小的时候
            bz_end = items_num - batch_size
            index = self.index[bz_end:self.end]#index part1
            index_part = self.index[0:items_num]#index part2
            index.extend(index_part)#List的连接操作
            self.front = items_num#update front cursor
        else:#当剩余元素数量大于batch_size
            if items_num == 0:
                self.reset_cursor()#重置游标
            bz_end = self.front + batch_size
            index = self.index[self.front:bz_end]# index section
            self.front = bz_end#update front cursor
        batch = []
        #generate batch
        for i in range(len(self.m_items)):
            batch_part = []
            items = self.m_items[i]# get the itmes
            for j in range(len(index)):
                elem = items[index[j]]# get the elem
                batch_part.append(elem)
            batch.append(batch_part)
        return tuple(batch)
