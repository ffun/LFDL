#!/usr/bin/python
# -*- coding: UTF-8 -*-
import tensorflow as tf
from ffun.NetHelper import*
from ffun.LayerHelper import*
import ffun_data

class ffunNet(object):
    'ffun-net for LFDL'
    def __init__(self, lr, skip_layer=None, weights_path=None):
        '''
        Inputs:
        - skip_layer: list of strings, names of the layers you want to reinitialize
        - weights_path: path string, path to the pretrained weights
        '''
        # Net param
        self.SKIP_LAYER = skip_layer
        self.WEIGHTS_PATH = weights_path
        self.LR = lr
        self.IMAGES_PL = None
        self.KEEP_PROP_PL = None
        self.LABELS_PL = None
        # Net op
        self.Train_OP = None
        self.Loss_OP = None
        self.Eval_OP = None
        self.Infer_OP = None
        # hold a NetBuilder
        self.builder = TfBuilder()
    def __check(self):
        'check if the Net is builder'
        if self.Infer_OP is None:
            raise Exception('Please create or load Net first')
    def build(self, images_pl, labels_pl, keep_prop_pl):
        'create infer op'
        return self.infer(images_pl, labels_pl, keep_prop_pl)
    def infer(self, images_pl, labels_pl, keep_prop_pl):
        '''
        funtion to creat infer op for Net
        Input:
        - images_pl:placeholder for images
        - labels_pl:placeholder for labels
        - keep_prop_pl:placeholder for keep_prop
        '''
        if self.Infer_OP is not None:
            return self.Infer_OP
        #hold the placeholder
        self.KEEP_PROP_PL = keep_prop_pl
        self.IMAGES_PL = images_pl
        self.LABELS_PL = labels_pl
        builder = self.builder
        # conv layer
        conv1 = builder.conv2d(
            images_pl,
            [3, 3, 3, 64],
            [1, 1, 1, 1],
            'VALID',
            name='conv1',
            initializer_w=tf.truncated_normal_initializer(mean=0.0, stddev=1e-2),
            initializer_b=tf.truncated_normal_initializer(mean=0.0, stddev=1e-2)
        )
        # pool layer
        pool1 = builder.max_pool(conv1, [1, 1, 2, 1], [1, 1, 2, 1], 'VALID', 'pool1')
        # conv layer
        conv2 = builder.conv2d(
            pool1, [3, 3, 64, 128], [1, 1, 1, 1], 'VALID', 'conv2',
            initializer_w=tf.truncated_normal_initializer(mean=0.0, stddev=1e-2),
            initializer_b=tf.truncated_normal_initializer(mean=0.0, stddev=1e-2)
            )
        # pool layer
        pool2 = builder.max_pool(conv2, [1, 1, 2, 1], [1, 1, 2, 1], 'VALID', 'pool2')
        # fc layer
        flattened = tf.reshape(pool2, [-1, 5*6*128])
        fc1 = builder.fc(
            flattened, [5*6*128, 1024], 'fc1',
        )
        dropout1 = builder.dropout(fc1, keep_prop_pl)
        # fc layer. Disable relu for loss funtion
        fc2 = builder.fc(dropout1, [1024, 58], 'fc2', relu=False)
        #assign the fc2 to scode
        self.Infer_OP = fc2
        return self.Infer_OP
    def loss(self):
        'loss op'
        self.__check()
        #get Loss op
        if self.Loss_OP is not None:
            return self.Loss_OP
        # 分类的softmaxLoss,采用信息熵形式
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=self.LABELS_PL, logits=self.Infer_OP, name='xentropy')
        self.Loss_OP = tf.reduce_mean(cross_entropy)
        return self.Loss_OP
    def train(self):
        'get Train op'
        self.__check()
        if self.Train_OP is not None:
            return self.Train_OP
        optimizer = tf.train.AdamOptimizer(self.LR, 0.9, 0.995)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op = optimizer.minimize(self.loss(), global_step=global_step)
        self.Train_OP = train_op
        return self.Train_OP
    def eval(self):
        'get Eval op'
        self.__check()
        if self.Eval_OP is not None:
            return self.Eval_OP
        correct = tf.nn.in_top_k(self.Infer_OP, self.LABELS_PL, 1)
        accuracy = tf.reduce_sum(tf.cast(correct, tf.int32))
        self.Eval_OP = accuracy
        return self.Eval_OP
    def run_train(self, sess, dataset):
        'do train for Net'
        feed_dict = ffun_data.fill_feed_dict(
            #dataset.next_batch(), images_pl, labels_pl, prop_pl, mode='train'
            dataset.next_batch(), self.IMAGES_PL, self.LABELS_PL, self.KEEP_PROP_PL, mode='train'
        )
        _, loss = sess.run([self.train(), self.loss()], feed_dict=feed_dict)
        return loss
    def run_eval(self, sess, dataset):
        'do eval for Net'
        true_count = 0.0
        step_epochs = dataset.num() // dataset.batch_size()
        for step in xrange(step_epochs):
            #get feed_dict
            feed_dict = ffun_data.fill_feed_dict(
                dataset.next_batch(), self.IMAGES_PL, self.LABELS_PL, self.KEEP_PROP_PL, mode='test'
            )
            #分类准确率计算
            true_count += sess.run(self.eval(), feed_dict=feed_dict)
        #准确率
        precision = float(true_count)/dataset.num()
        eval_info = 'num_examples:%d,correct:%d,precision:%0.04f'
        eval_info = eval_info % (dataset.num(), true_count, precision)
        return eval_info
    def load_weights(self):
        'load pretrained model'
        pass
