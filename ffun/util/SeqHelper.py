#!/usr/bin/python
# -*- coding: UTF-8 -*-
import math
class SeqHelper(object):
    @classmethod
    def stat_2seq(cls, seq1, seq2, prec):
        '''
        统计两个浮点序列中，对应元素的误差在精度prec之内的元素个数
        '''
        num=0
        for i in range(len(seq1)):
            if math.fabs(seq1[i], seq2[i]) < prec:
                num=num+1
        return num