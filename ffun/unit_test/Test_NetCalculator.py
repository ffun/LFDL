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
#print messge
nc.print_layers()
print nc.num_of_layers()