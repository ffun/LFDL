#!/usr/bin/python
# -*- coding: UTF-8 -*-

import ffun.util as Fut
import ffun.io as Fio

'''
#example1:generate the origin epi files
files =  Fio.FileHelper.get_files('/Users/fang/workspaces/tf_space/box')
Epi_creator = Fio.EPIcreator(files)
Epi_creator.create((45,53))
'''
'''
#example2:get labels,return a matrix of 512x512
#every element of the matrix is a float
labelloader = Fio.TextLoader()
labelloader.read('/Users/fang/workspaces/tf_space/LFDL/disp.txt', float)
'''

#example3:extract epi file,return a numpy ndarray
extractor = Fio.EPIextractor('/Users/fang/workspaces/tf_space/Box/epi45_53/epi_45_53_002.png')
extractor.set_padding(0,'1')

'''
#example4
batch = [1,2,3,4,5]
label = [1,2,3,4,5]
bh = Fut.BatchHelper((batch,label))
bh.shuffle()
current = bh.head()
#judge if current is None
if current != None:
    d = current[0]
    l = current[1]
    print d,l
'''