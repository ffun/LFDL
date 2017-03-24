#!/usr/bin/python
# -*- coding: UTF-8 -*-
'''
实际运行该单元测试时，需要放到具体模块下
'''
from BatchHelper import BatchHelper
from Checker import Checker
a = [1,2,3,4,5,6,7]
b = [1,2,3,4,5,6,7]

bh = BatchHelper((a, b))

bh.shuffle(5)
'''
Checker.seq_len_check(bh.get_batch(5)[0], 5)
Checker.seq_len_check(bh.get_batch(5)[0], 5)

print 'Test:BatchHelper OK'
'''
bh.print_data()
for i in range(5):
    print bh.next_batch(5)
