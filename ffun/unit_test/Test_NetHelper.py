#!/usr/bin/python
# -*- coding: UTF-8 -*-
import NetHelper

L1 = NetHelper.Data_Layer([9, 33, 3])
L2 = NetHelper.Conv_Layer([3, 3, 3, 64])
L3 = NetHelper.Pool_Layer([1, 1, 2, 1], [1, 1, 2, 1])
L4 = NetHelper.Conv_Layer([3, 3, 64, 128])
L5 = NetHelper.Pool_Layer([1, 1, 2, 1], [1, 1, 2, 1])
L6 = NetHelper.Fc_Layer([128*5*6, 1024])
L7 = NetHelper.Fc_Layer([1024, 58])
net = NetHelper.Net(L1, L2, L3, L4, L5, L6, L7)
#或者通过net.add_layer(L1)添加层

print net.info()
print 'layer num:', net.layer_num()
print 'weight_memery_cost:', net.weight_memery_cost()
print 'hidden_memory_cost:', net.hidden_memory_cost()
print 'data_memory_cost:', net.data_memory_cost()
print 'all_memort_cost:', net.all_memory_cost()

