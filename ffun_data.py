#!/usr/bin/python
# -*- coding: UTF-8 -*-
'''
本模块主要是为了给网络生成epi文件，以及提供data-batch
'''
import tensorflow as tf
import ffun.util as Fut
import ffun.io as Fio
import numpy as np

#配置文件
train_data_cfg = {
    'img-dir':'/Users/fang/workspaces/tf_space/box',
    'origin-epi-dir':'/Users/fang/workspaces/tf_space/box/epi45_53',
    'label-dir':'/Users/fang/workspaces/tf_space/LFDL/disp.txt'
}
#img 配置文件
img_cfg = {
    'height':9,
    'width':512,
    'channel':3
}
#epi 配置文件
epi_cfg = {
    'height':9,
    'width':33,#需要提取的epi长度
    'channel':3,
    'mode':'c=0'
}

#generate origin epi data
def epi_data_generate():
    files =  Fio.FileHelper.get_files(train_data_cfg['img-dir'])
    Epi_creator = Fio.EPIcreator(files)
    Epi_creator.create((45, 53))

#batch-data
def batch_data(tdc=train_data_cfg):
    '''
    function:to generante batch Object for use next_batch() API\n
    @tdc:train data configure object
    '''
    #得到排序后的图片文件列表
    origin_epi_list = Fio.FileHelper.get_files(tdc['origin-epi-dir'])
    epi_list = []
    for i in range(len(origin_epi_list)):
        Extractor = Fio.EPIextractor(origin_epi_list[i])
        #给原图像加上padding,这样下面我们就可以提取长度为33的
        Extractor.set_padding(epi_cfg['width']/2, epi_cfg['mode'])
        for j in range(img_cfg['width']):
            epi = Extractor.extract(j, epi_cfg['width'])
            epi_list.append(epi)
    #load labels
    LabelLoader = Fio.TextLoader()
    labels = LabelLoader.read(train_data_cfg['label-dir'], float)
    labels = np.array(labels)#转为numpy.ndarray类型数据
    assert len(labels) == len(epi_list)#check
    data_batch = Fut.BatchHelper((epi_list, labels))
    print 'generate batch done!'
    return data_batch

def placeholder_inputs(batch_size):
    '''
    function to generate placeholder

    Args:
        batch_size: The batch size will be baked into both placeholders.

    Returns:
        images_placeholder: Images placeholder.
        labels_placeholder: Labels placeholder.
    '''
    H, W, C = epi_cfg['height'], epi_cfg['width'], epi_cfg['channel']
    epi_img_placeholder = tf.placeholder(tf.float32, shape=(batch_size, H, W, C))
    labels_placeholder = tf.placeholder(tf.float32, shape=(batch_size))
    keep_prob_placeholder = tf.placeholder('float')
    return epi_img_placeholder, labels_placeholder, keep_prob_placeholder

def fill_feed_dict(data_set, images_pl, labels_pl, prob_pl, mode='train'):
    '''
    function to generate feed_dict\n
    @data_set:The set of images and labels,from batch_data()
    @images_pl: The images placeholder,from placeholder_inputs().
    @labels_pl: The labels placeholder,from placeholder_inputs().
    @prob_pl: The keep_prop placeholder,from placeholder_inputs().
    '''
    data = data_set
    images_feed, labels_feed = data[0], data[1]
    prop = 0.5
    if cmp(mode, 'test'):#if test phase
        prop = 1.0
    feed_dict = {
      images_pl: images_feed,
      labels_pl: labels_feed,
      prob_pl: prop
    }
    return feed_dict

if __name__ == '__main__':
    epi_data_generate()
