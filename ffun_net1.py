#!/usr/bin/python
# -*- coding: UTF-8 -*-
import ffun.util as Fut
import tensorflow as tf
import ffun_data

class ffunNet(object):
    def __init__(self, x, keep_prop, num_of_classes, skip_layer, weights_path=None):
        '''
        Inputs:
        - x: tf.placeholder, for the input images
        - keep_prob: tf.placeholder, for the dropout rate
        - skip_layer: list of strings, names of the layers you want to reinitialize
        - weights_path: path string, path to the pretrained weights
        '''
        self.INPUT = x
        self.KEEP_PROP = keep_prop
        self.NUM_OF_CLASSES = num_of_classes
        self.SKIP_LAYER = skip_layer
        self.WEIGHTS_PATH = weights_path
        # Net op
        self.Train_op = None
        self.Loss_op = None
        self.Eval_op = None
        self.Inference = None
        # hold a NetBuilder
        self.builder = Fut.TfBuilder()
        self.create()
    def create(self, Phase='train'):
        '''
        funtion to creat Net for train  
        Input:
        - Phase:'train'/'test' for keep_prop
        '''
        builder = self.builder
        # conv layer
        conv1 = builder.conv2d(self.Inference, [3, 3, 3, 64], [1, 1, 1, 1], 'VALID', name='conv1')
        # pool layer
        pool1 = builder.max_pool(conv1, [1, 1, 2, 1], [1, 1, 1, 1], 'VALID', 'pool1')
        # conv layer
        conv2 = builder.conv2d(pool1, [3, 3, 64, 128], [1, 1, 1, 1], 'VALID', 'conv2')
        # pool layer
        pool2 = builder.max_pool(conv2, [1, 1, 2, 1], [1, 1, 1, 1], 'VALID', 'pool2')
        # fc layer
        flattened = tf.reshape(pool2, [-1, 5*6*128])
        fc1 = builder.fc(flattened, [5*6*128, 1024], 'fc1')
        dropout1 = builder.dropout(fc1, self.KEEP_PROP)
        # fc layer. Disable relu for loss funtion
        fc2 = builder.fc(dropout1, [1024, self.NUM_OF_CLASSES], 'fc2', relu=False)
        #assign the fc2 to scode
        self.Inference = fc2
    def infer(self):
        'get scode'
        return self.Inference
    def loss(self, labels):
        'loss op'
        #get Loss op
        if self.Loss_op:
            return self.Loss_op
        #分类的softmaxLoss,采用信息熵形式
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=self.Inference, name='xentropy')
        self.Loss_op = tf.reduce_mean(cross_entropy)
    def train(self, loss, lr):
        'get Train op'
        if self.Train_op:
            return self.Train_op
        optimizer = tf.train.AdamOptimizer(lr, 0.9, 0.995)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op = optimizer.minimize(loss, global_step=global_step)
        self.Train_op = train_op
    def eval(self, labels):
        'get Eval op'
        if self.Eval_op:
            return self.Eval_op
        correct = tf.nn.in_top_k(self.Inference, labels, 1)
        accuracy = tf.reduce_sum(tf.cast(correct, tf.int32))
        self.Eval_op = accuracy
    def run_train(self, sess, images_pl, labels_pl, prop_pl, dataset):
        '''
        funtion to train the net

        Input:
        - sess:TensorFlow's session object
        - images_pl:placeholder of images
        - labels_pl:placeholder of labels
        - prop_pl:placeholder of keep_prop

        Return:
        - loss value
        '''
        feed_dict = ffun_data.fill_feed_dict(
            dataset.next_batch(), images_pl, labels_pl, prop_pl, mode='train'
        )
        loss_op = self.loss(labels_pl)
        train_op = self.train(loss_op, 1e-4)
        _, loss = sess.run([train_op, loss_op], feed_dict=feed_dict)
        return loss
    def run_eval(self, sess, images_pl, labels_pl, prop_pl, dataset):
        '''
        funtion to eval the net

        Input:
        - sess:TensorFlow's session object
        - images_pl:placeholder of images
        - labels_pl:placeholder of labels
        - prop_pl:placeholder of keep_prop
        '''
        true_count = 0.0
        step_epochs = dataset.num() // dataset.batch_size()
        for step in xrange(step_epochs):
            #get feed_dict
            feed_dict = ffun_data.fill_feed_dict(
                dataset.next_batch(), images_pl, labels_pl, prop_pl, mode='test'
            )
            #分类准确率计算
            true_count += sess.run(self.eval(labels_pl), feed_dict=feed_dict)
        #准确率
        precision = float(true_count)/dataset.num()
        eval_info = 'num_examples:%d,correct:%d,precision:%0.04f'
        eval_info = eval_info % (dataset.num(), true_count, precision)
        print eval_info
    def load_initial_weights(self):
        pass