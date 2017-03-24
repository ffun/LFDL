#!/usr/bin/python
# -*- coding: UTF-8 -*-
import tensorflow as tf
import ffun.util as Fut
import ffun.io as Fio

def batch_data():
    img_info = {
        'height':9,
        'width':512,
        'channel':3
    }

    origin_epi_path = '/Users/fang/workspaces/tf_space/box/epi45_53'
    #得到排序后的图片文件列表
    origin_epi_list = Fio.FileHelper.get_files(origin_epi_path)

    epi_list = []
    epi_img_length = 33#epi提取的长度
    for i in range(len(origin_epi_list)):
        Extractor = Fio.EPIextractor(origin_epi_list[i])
        Extractor.set_padding(epi_img_length/2, 'c=0')#给原图像加上padding,这样下面我们就可以提取长度为33的
        for j in range(img_info['width']):
            epi_list.append(Extractor.extract(j, epi_img_length))
    #load labels
    label_path = '/Users/fang/workspaces/tf_space/LFDL/disp.txt'
    labels = Fio.TextLoader.read(label_path, float)

    assert len(labels) == len(epi_list)
    data_batch = Fut.BatchHelper((epi_list, labels))
    print 'generate batch done!'
    return data_batch
