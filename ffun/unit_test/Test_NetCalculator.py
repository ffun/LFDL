#!/usr/bin/python
# -*- coding: UTF-8 -*-

import ffun.util as Fut
#创建对象
nc = Fut.NetCalculator()
#第一步是设置数据层，否则会报错
nc.set_dataLayer([9,33,3])
#添加层信息
nc.add_conv_layer(ksize=[3, 3, 3, 64])
nc.add_pool_layer(ksize=[1,1,2,1],strides=[1,1,2,1])
nc.add_conv_layer(ksize=[3, 3, 64, 128])
nc.add_pool_layer(ksize=[1,1,2,1],strides=[1,1,2,1])
nc.add_fc_layer([128*5*6,1024])
#print messge
nc.print_layers()
print 'layer num:',nc.num_of_layers()
print 'weight_memery_cost:',nc.weight_memery_cost()
print 'hidden_memory_cost:',nc.hidden_memory_cost()
print 'data_memory_cost:',nc.data_memory_cost()
print 'all_memort_cost:',nc.all_memory_cost()
