#!/usr/bin/python
# -*- coding: UTF-8 -*-
import sys,getopt
import CFG
from ffun.EPI import*
from ffun.DataProvider import DataProvider,BatchHelper
from ffun.FileHelper import*
from ffun import LabelHelper
import numpy as np
import tensorflow as tf

def epi_generate():
    '生成单个样本的原始epi文件'
    print 'gengerating EPI Files'
    files = FileHelper.get_files(CFG.Image_DIR)
    epi_creator = EPIcreator(files)
    epi_creator.create((36, 44))

def epi_patch_generate():
    '生成用于训练的epi patch'
    pass

def label_trans(x, class_num=CFG.Class_NUM):
    '标签转换函数:把float转换成int，[0,57]'
    r = (x+2)*class_num/4
    r = int(round(r))#四舍五入后取整
    if r > class_num:
        r = class_num -1
    elif r < 0:
        r = 0
    return r

def class_trains(y, class_num = CFG.Class_NUM):
    '类别转换函数：把infer得到的class转换成float以便与label比较'
    x = y*4.0/class_num - 2
    return x

#batch-data
def get_data():
    '''
    function:to get data\n
    return:BatchHelper OBJ
    '''
    #load labels
    LabelLoader = LabelHelper()
    #labels = LabelLoader.read(train_file_cfg['label-dir'], float)
    labels = LabelLoader.read(CFG.Label_DIR, float)
    #对label进行偏移,label->[-2,2],new_labels->[0,4],并且进行离散化成58个类
    labels = map(label_trans, labels)#make it 58 classes between [0,4]
    print 'load labels done!'
    labels = np.array(labels)#转为numpy.ndarray类型数据
    #得到排序后的图片文件列表
    #origin_epi_list = Fio.FileHelper.get_files(tdc['origin-epi-dir'])
    origin_epi_list = FileHelper.get_files(CFG.EPI_DIR)
    epi_list = []
    for i in xrange(len(origin_epi_list)):
        Extractor = EPIextractor(origin_epi_list[i])
        #给原图像加上padding,这样下面我们就可以提取长度为33的
        Extractor.set_padding(CFG.Input_W)
        #for j in xrange(img_cfg['width']):
        for j in xrange(CFG.EPI_W):
            epi = Extractor.extract(j, CFG.Input_W)
            epi_list.append(epi)
    print 'generate epi done!'
    assert len(labels) == len(epi_list)#check
    print 'generate data done!'
    return BatchHelper((epi_list, labels))

def get_data_bh():
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
    train_bh = BatchHelper((images[0:num_tr], labels[0:num_tr]))
    verify_bh = BatchHelper((images[num_tr:num_vf], labels[num_tr:num_vf]))
    test_bh = BatchHelper((images[num_vf:num_te], labels[num_vf:num_te]))
    return train_bh, verify_bh, test_bh

class DataSource(DataProvider):
    'provide data、palceholder and feed_dict'
    def __init__(self, bh=None, batch_size=50, mode='once'):
        '''
        Input:
        - mode:'once'数据内容一次加载至内存，'part':分步加载至内存
        '''
        super(DataSource, self).__init__(bh, batch_size, mode)
        self.IMAGES_PL = None
        self.KEEP_PROP_PL = None
        self.LABELS_PL = None
        self.PL_OK = False
    def get_placeholder(self):
        '获得palceholder'
        if self.PL_OK:
            return self.IMAGES_PL, self.LABELS_PL, self.KEEP_PROP_PL
        H, W, C = CFG.Input_H, CFG.Input_W, CFG.Input_C
        self.IMAGES_PL = tf.placeholder(tf.float32, shape=(self.batch_size(), H, W, C))
        self.LABELS_PL = tf.placeholder(tf.int32, shape=(self.batch_size()))
        self.KEEP_PROP_PL = tf.placeholder('float')
        self.PL_OK = True
        return self.IMAGES_PL, self.LABELS_PL, self.KEEP_PROP_PL
    def get_feeddict(self, mode='train'):
        '获得feeddict'
        self.get_placeholder()
        prop = 0.5
        if mode == 'test':
            prop = 1.0
        images_feed, labels_feed = self.next_batch()#获得数据
        feed_dict = {
            self.IMAGES_PL: images_feed,
            self.LABELS_PL: labels_feed,
            self.KEEP_PROP_PL: prop
        }
        return feed_dict

class TestData(DataSource):
    '测试数据集提供者'
    def get_placeholder(self):
        '获得palceholder'
        if self.PL_OK:
            return self.IMAGES_PL, self.LABELS_PL, self.KEEP_PROP_PL
        H, W, C = CFG.Input_H, CFG.Input_W, CFG.Input_C
        self.IMAGES_PL = tf.placeholder(tf.float32, shape=(self.batch_size(), H, W, C))
        #原始数据集，采用tf.float32作为占位符
        self.LABELS_PL = tf.placeholder(tf.float32, shape=(self.batch_size()))
        self.KEEP_PROP_PL = tf.placeholder('float')
        self.PL_OK = True
        return self.IMAGES_PL, self.LABELS_PL, self.KEEP_PROP_PL
    def get_feeddict(self):
        '获得feeddict'
        self.get_placeholder()
        prop = 1.0
        images_feed, labels_feed = self.next_batch()#获得数据
        feed_dict = {
            self.IMAGES_PL: images_feed,
            self.LABELS_PL: labels_feed,
            self.KEEP_PROP_PL: prop
        }
        return feed_dict


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
