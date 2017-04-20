#!/usr/bin/python
# -*- coding: UTF-8 -*-
'''
file:BatchHelper.py\n
@author:fang.junpeng\n
@email:tfzsll@126.com
'''

import random
import sys
from FileHelper import FileHelper,LabelHelper
from DataUtil import ImageHelper

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
        self.index = range(len(self.m_items[0]))#generate a list
        self.front = 0
        self.end = len(m_items[0])#self.end point to the end of seq
    def __check(self):
        '''
        数据校验
        '''
        #类型校验
        assert isinstance(self.m_items, tuple) or isinstance(self.m_items, list)
        #长度校验
        seq_length = len(self.m_items[0])
        for i in xrange(len(self.m_items)):
            assert len(self.m_items[i]) == seq_length

    def shuffle(self, times=1):
        '''
        shuffle the m_items
        '''
        self.reset_cursor()
        for _ in xrange(times):
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
            for i in xrange(len(self.m_items)):
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
        for i in xrange(len(self.m_items)):
            item = self.m_items[i]
            for j in xrange(len(index)):
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
        for i in xrange(len(self.m_items)):
            data_part = []
            item = self.m_items[i]
            for j in xrange(len(index)):
                elem = item[index[j]]
                data_part.append(elem)
            data.append(data_part)
        return data
    def next_batch(self, batch_size):
        '''
        function to get a batch items of batch-size\n
        it will get batch circularly\n
        suggestion:batch_size should be less than which the Object holds
        '''
        items_num = self.end - self.front#剩余元素的总数
        bz_end = items_num
        index = None
        #set bz_end circularly && generate index
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
        for i in xrange(len(self.m_items)):
            batch_part = []
            items = self.m_items[i]# get the itmes
            for j in xrange(len(index)):
                elem = items[index[j]]# get the elem
                batch_part.append(elem)
            batch.append(batch_part)
        return tuple(batch)

    def num(self):
        '获得BH所持有的数据个数'
        return len(self.index)
    def num_remain(self):
        '获得BH剩余数据个数:由于BH支持循环读取，所以此值仅代表本轮的remain'
        return self.end - self.front

class DataProvider(object):
    'ffun package的数据组件'
    def __init__(self, bh=None, batch_size=50, mode='once'):
        '''
        Input:
        - mode:'once'数据内容一次加载至内存，'part':分步加载至内存
        '''
        self.BH = bh#所持有的BH，如果是once模式，BH持有数据和标签；否则BH持有数据路径和标签
        self.BZ = (batch_size,)
        self.MODE = (mode,)#使用元组形式存储MODE，防止被修改
    def load_from_files(self, data_dir, suffix, label_path, transform):
        '''
        加载文件路径和标签数据并封装成BH存储
        '''
        files = FileHelper.get_files(data_dir, suffix)
        labels = LabelHelper.read(label_path, transform)
        assert len(files) == len(labels)
        self.BH = BatchHelper((files, labels))
    def load_from_BH(self, bh):
        '加载BH'
        if not isinstance(bh, BatchHelper):
            raise TypeError('bh should be BatchHelper OBJ')
        self.BH = bh
    def next_batch(self):
        'get next batch data'
        bz = self.BZ[0]
        if self.MODE[0] == 'once':
            return self.BH.next_batch(bz)
        elif self.MODE[0] == 'part':
            '''
            1.在分布加载模型中，是读取self.BH中存储的文件路径的文件，然后组成新的BH并获取next_batch
            2.默认在part模式，每次加载batch-size个数据，为了加速，其实一次性可以加载bs的n倍数据.
            但是此时这种情况下，需要让该类持有一个额外的bh临时保存，每次先检查它的num_remain()值，如果为0了，
            再加载一次。同时要考虑好n的大小，让n*bs不大于BH.num()
            '''
            paths, labels = self.BH.next_batch(bz)
            data = []
            for path in paths:
                data.append(ImageHelper().read(path).data_convert3d())
            return BatchHelper((data, labels)).next_batch(bz)
    def num(self):
        '获得所持有的数据个数总数'
        return self.BH.num()
    def batch_size(self):
        '获取batch-size的值'
        return self.BZ[0]
