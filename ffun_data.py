#!/usr/bin/python
# -*- coding: UTF-8 -*-
'''
本模块主要是为了给网络生成epi文件，以及提供data-batch
'''
import tensorflow as tf
import ffun.util as Fut
import ffun.io as Fio

ffun_data_cfg = {
    'img-dir':'/Users/fang/workspaces/tf_space/box',
    'origin-epi-dir':'/Users/fang/workspaces/tf_space/box/epi45_53',
    'label-dir':'/Users/fang/workspaces/tf_space/LFDL/disp.txt'
}

#generate origin epi data
def epi_data_generate():
    files =  Fio.FileHelper.get_files(ffun_data_cfg['img-dir'])
    Epi_creator = Fio.EPIcreator(files)
    Epi_creator.create((45, 53))

#batch-data
def batch_data():
    #img 配置文件
    img_cfg = {
        'height':9,
        'width':512,
        'channel':3
    }
    #epi 配置文件
    epi_cfg = {
        'length':33,#需要提取的epi长度
        'mode':'c=0'
    }
    #得到排序后的图片文件列表
    origin_epi_list = Fio.FileHelper.get_files(ffun_data_cfg['origin-epi-dir'])
    epi_list = []
    for i in range(len(origin_epi_list)):
        Extractor = Fio.EPIextractor(origin_epi_list[i])
        #给原图像加上padding,这样下面我们就可以提取长度为33的
        Extractor.set_padding(epi_cfg['length']/2, epi_cfg['mode'])
        for j in range(img_cfg['width']):
            epi_list.append(Extractor.extract(j, epi_cfg['length']))
    #load labels
    labels = Fio.TextLoader.read(ffun_data_cfg['label-dir'], float)
    assert len(labels) == len(epi_list)#check
    data_batch = Fut.BatchHelper((epi_list, labels))
    print 'generate batch done!'
    return data_batch

if __name__ == '__main__':
    epi_data_generate()
