#!/usr/bin/python
# -*- coding: UTF-8 -*-
'''
本模块主要是为了给网络生成epi文件，以及提供data-batch
'''
import tensorflow as tf
import ffun.util as Fut
import ffun.io as Fio
import numpy as np
import sys, getopt

#配置文件
train_file_cfg = {
    'img-dir':'/Users/fang/workspaces/tf_space/box',
    'origin-epi-dir':'/Users/fang/workspaces/tf_space/box/epi36_44',
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
#配置文件:训练、验证、测试集的数量
data_cfg = {
    'all':512*512,
    'train':200000,
    'verify':40000,
    'test':22000
}

#generate origin epi data
def epi_data_generate():
    print 'Start gengerating EPI Files'
    files =  Fio.FileHelper.get_files(train_file_cfg['img-dir'])
    Epi_creator = Fio.EPIcreator(files)
    Epi_creator.create((36, 44))

def label_trans(x, class_num=58):
    '''
    function to transform float label to int
    '''
    r = (x+2)*class_num/4
    r = int(r)
    if r > class_num:
        r = class_num
    elif r < 0:
        r = 0
    return r

#batch-data
def get_data(tdc=train_file_cfg):
    '''
    function:to get data\n
    @tdc:train data configure object\n
    return:BatchHelper obj
    '''
    #load labels
    LabelLoader = Fio.TextLoader()
    labels = LabelLoader.read(train_file_cfg['label-dir'], float)
    #对label进行偏移,label->[-2,2],new_labels->[0,4],并且进行离散化成58个类
    labels = map(label_trans, labels)#make it 58 classes between [0,4]
    print 'load labels done!'
    labels = np.array(labels)#转为numpy.ndarray类型数据
    #得到排序后的图片文件列表
    origin_epi_list = Fio.FileHelper.get_files(tdc['origin-epi-dir'])
    epi_list = []
    for i in xrange(len(origin_epi_list)):
        Extractor = Fio.EPIextractor(origin_epi_list[i])
        #给原图像加上padding,这样下面我们就可以提取长度为33的
        Extractor.set_padding(epi_cfg['width']/2, epi_cfg['mode'])
        for j in xrange(img_cfg['width']):
            epi = Extractor.extract(j, epi_cfg['width'])
            epi_list.append(epi)
    print 'generate epi done!'
    assert len(labels) == len(epi_list)#check
    print 'generate data done!'
    return Fut.BatchHelper((epi_list, labels))

def data():
    '''
    provide data
    '''
    bh = get_data()
    bh.shuffle(10)#乱序10次
    new_data = bh.get_data()
    images, labels = new_data[0], new_data[1]
    num_tr = data_cfg['train']
    num_vf = data_cfg['verify']+num_tr
    num_te = data_cfg['test']+num_vf
    train_bh = Fut.BatchHelper((images[0:num_tr], labels[0:num_tr]))
    verify_bh = Fut.BatchHelper((images[num_tr:num_vf], labels[num_tr:num_vf]))
    test_bh = Fut.BatchHelper((images[num_vf:num_te], labels[num_vf:num_te]))
    return train_bh, verify_bh, test_bh

def placeholder_inputs(batch_size):
    '''
    function to generate placeholder

    Args:
        batch_size: The batch size will be baked into both placeholders.

    Returns:
        images_pl: Images placeholder.
        labels_pl: Labels placeholder.
    '''
    H, W, C = epi_cfg['height'], epi_cfg['width'], epi_cfg['channel']
    images_pl = tf.placeholder(tf.float32, shape=(batch_size, H, W, C))
    '''
    # 回归任务
    labels_pl = tf.placeholder(tf.float32, shape=(batch_size))
    '''
    # 分类任务
    labels_pl = tf.placeholder(tf.int32, shape=(batch_size))
    keep_prob_pl = tf.placeholder('float')
    return images_pl, labels_pl, keep_prob_pl

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
    if mode == 'test':#if test phase
        prop = 1.0
    feed_dict = {
      images_pl: images_feed,
      labels_pl: labels_feed,
      prob_pl: prop
    }
    return feed_dict

def usage():
    nl = '\n'
    usage_str = 'OVERVIEW: ffun_data tool'+nl
    usage_str += nl + 'usage:' +nl
    usage_str += 'ffun_data.py [options] <inputs>'+nl
    usage_str += nl+'OPTIONS:'+nl
    usage_str += '-h    >>> to get help of this program\n'
    usage_str += '--epi >>> to generate origin epi files'
    print usage_str

if __name__ == '__main__':
    opts, _ = getopt.getopt(sys.argv[1:], "h", ['epi'])
    #epi_data_generate()
    for op, value in opts:
        if op == '--epi':
            epi_data_generate()
        elif op == '-h':
            usage()
            sys.exit()
    if len(opts) == 0:
        usage()
