#!/usr/bin/python
# -*- coding: UTF-8 -*-
import sys,getopt
import CFG
import ffun.io as Fio
import ffun.util as Fut
import numpy as np
import tensorflow as tf

def epi_generate():
    '生成单个样本的原始epi文件'
    print 'gengerating EPI Files'
    files = Fio.FileHelper.get_files(CFG.Image_DIR)
    epi_creator = Fio.EPIcreator(files)
    epi_creator.create((36, 44))

def epi_patch_generate():
    '生成用于训练的epi patch'
    pass

def label_trans(x, class_num=58):
    '标签转换函数:把float转换成int，[0,57]'
    r = (x+2)*class_num/4
    r = int(round(r))#四舍五入后取整
    if r > class_num:
        r = class_num -1
    elif r < 0:
        r = 0
    return r

#batch-data
def get_data():
    '''
    function:to get data\n
    return:BatchHelper OBJ
    '''
    #load labels
    LabelLoader = Fio.TextLoader()
    #labels = LabelLoader.read(train_file_cfg['label-dir'], float)
    labels = LabelLoader.read(CFG.Label_DIR, float)
    #对label进行偏移,label->[-2,2],new_labels->[0,4],并且进行离散化成58个类
    labels = map(label_trans, labels)#make it 58 classes between [0,4]
    print 'load labels done!'
    labels = np.array(labels)#转为numpy.ndarray类型数据
    #得到排序后的图片文件列表
    #origin_epi_list = Fio.FileHelper.get_files(tdc['origin-epi-dir'])
    origin_epi_list = Fio.FileHelper.get_files(CFG.EPI_DIR)
    epi_list = []
    for i in xrange(len(origin_epi_list)):
        Extractor = Fio.EPIextractor(origin_epi_list[i])
        #给原图像加上padding,这样下面我们就可以提取长度为33的
        Extractor.set_padding(CFG.Input_W)
        #for j in xrange(img_cfg['width']):
        for j in xrange(CFG.EPI_W):
            epi = Extractor.extract(j, CFG.Input_W)
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
    num_tr = CFG.Data_TRAIN_NUM
    num_vf = num_tr + CFG.Data_VERIFY_NUM
    num_te = num_vf + CFG.Data_TEST_NUM
    train_bh = Fut.BatchHelper((images[0:num_tr], labels[0:num_tr]))
    verify_bh = Fut.BatchHelper((images[num_tr:num_vf], labels[num_tr:num_vf]))
    test_bh = Fut.BatchHelper((images[num_vf:num_te], labels[num_vf:num_te]))
    return train_bh, verify_bh, test_bh

class DataProvider(object):
    def __init__(self, batch_size=1):
        self.TRAIN_DATA = None
        self.VERIFY_DATA = None
        self.TEST_DATA = None
        self.BATCH_SIZE = batch_size
    def load_files(self, data_dir, label_path):
        pass
    def funcname(self, parameter_list):
        pass
    def get_train_data(self):
        pass
    def get_verify_data(self):
        pass
    def get_test_data(self):
        pass
    def get_data(self):
        pass

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


def placeholder_inputs(batch_size):
    '''
    function to generate placeholder
    Args:
        batch_size: The batch size will be baked into both placeholders.
    Returns:
        images_pl: Images placeholder.
        labels_pl: Labels placeholder.
    '''
    H, W, C = CFG.Input_H, CFG.Input_W, CFG.Input_C
    images_pl = tf.placeholder(tf.float32, shape=(batch_size, H, W, C))
    # 分类任务
    labels_pl = tf.placeholder(tf.int32, shape=(batch_size))
    keep_prob_pl = tf.placeholder('float')
    return images_pl, labels_pl, keep_prob_pl

def usage():
    '使用说明'
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
    for op, value in opts:
        if op == '--epi':
            epi_generate()
        elif op == '-h':
            usage()
    if len(opts) == 0:
        usage()
