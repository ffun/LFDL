#!/usr/bin/python
# -*- coding: UTF-8 -*-
import math
class SeqHelper(object):
    '''
    class for easy do with seq
    '''
    @classmethod
    def stat_2seq(cls, seq1, seq2, prec):
        '''
        统计两个浮点序列中，对应元素的误差在精度prec之内的元素个数
        '''
        num = 0
        for i in xrange(len(seq1)):
            if math.fabs(seq1[i] - seq2[i]) < prec:
                num = num + 1
        return num
    @classmethod
    def stat_seq(cls, seq, prec):
        num = 0
        for i in xrange(len(seq)):
            if seq[i] < prec:
                num += 1
        return num
