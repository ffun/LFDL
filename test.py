#!/usr/bin/python
# -*- coding: UTF-8 -*-

import ffun.util as Fut
import ffun.io as Fio

'''
#example1:generate the origin epi files
files =  Fio.FileHelper.get_files('/Users/fang/workspaces/tf_space/LFDL/pngdata')
Epi_creator = Fio.EPIcreator(files)
Epi_creator.create((36,44))

#example2:get labels,return a matrix of 512x512
#every element of the matrix is a float
labelloader = Fio.TextLoader()
labelloader.read('/Users/fang/workspaces/tf_space/LFDL/disp.txt', float)
'''
'''
#example3:extract epi file,return a numpy ndarray
extractor = Fio.EPIextractor('/Users/fang/workspaces/tf_space/LFDL/pngdata/epi36_44/epi_36_44_001.png')
extractor.extract(100)
'''
#example4
batch = [1,2,3,4,5]
label = [1,2,3,4,5]
bh = Fut.BatchHelper((batch,label))
bh.shuffle()
current_data = bh.head()[0]
current_label = bh.head()[1]

print current_data,current_label