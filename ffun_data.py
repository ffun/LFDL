#!/usr/bin/python
# -*- coding: UTF-8 -*-
'''
本模块主要是为了给网络生成epi文件，以及提供data-batch
'''
import tensorflow as tf
import ffun.util as Fut
import ffun.io as Fio

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
    'length':33,#需要提取的epi长度
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
        Extractor.set_padding(epi_cfg['length']/2, epi_cfg['mode'])
        for j in range(img_cfg['width']):
            epi = Extractor.extract(j, epi_cfg['length'])
            epi_list.append(epi)
    #load labels
    labels = Fio.TextLoader.read(train_data_cfg['label-dir'], float)
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
    H, W, C = img_cfg['height'], img_cfg['width'], img_cfg['channel']
    images_placeholder = tf.placeholder(tf.float32, shape=(batch_size, H, W, C))
    labels_placeholder = tf.placeholder(tf.int32, shape=(batch_size))
    return images_placeholder, labels_placeholder

def fill_feed_dict(data_set, images_pl, labels_pl):
    '''
    function to generate feed_dict\n
    @data_set:The set of images and labels,from batch_data()
    @images_pl: The images placeholder,from placeholder_inputs().
    @labels_pl: The labels placeholder,from placeholder_inputs().
    '''
    data = None
    if isinstance(data, Fut.BatchHelper):
        data = data_set.next_batch(50)
    images_feed, labels_feed = data[0],data[1]
    feed_dict = {
      images_pl: images_feed,
      labels_pl: labels_feed,
    }
    return feed_dict

if __name__ == '__main__':
    epi_data_generate()
