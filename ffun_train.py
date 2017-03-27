#!/usr/bin/python
# -*- coding: UTF-8 -*-
import tensorflow as tf
import os.path
import time
from ffun.util import *
import ffun_net
import ffun_data
'''
train_cfg={
    'model-dir':'/Users/fang/workspaces/tf_space/model',
    'train-batch':50,
    'test-batch':50
}'''

class Train_cfg(object):
    model_dir = '/Users/fang/workspaces/tf_space/model'
    batch_size = 50
class Test_cfg(object):
    batch_size = 50
    num_examples = 1000

def run_eval(sess, eval_correct, images_pl, labels_pl, dataset):
    
    pass

def run_train(max_steps):
    '''
    run train
    '''
    #get data
    train_bh, verify_bh, test_bh = ffun_data.data()
    #graph
    with tf.Graph().as_default():
        #get placeholder
        images_placeholder, labels_placeholder, keep_prop_placeholder = \
        ffun_data.placeholder_inputs(Train_cfg.batch_size)
        #get inference
        inference = ffun_net.infer(images_placeholder, keep_prop_placeholder)
        #get loss
        loss = ffun_net.loss(inference, labels_placeholder)
        #get train op
        train_op = ffun_net.train(loss, 1e-3)
        #init op
        init_op = tf.global_variables_initializer()
        #saver
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(init_op)
            start_time = time.time()
            for step in xrange(max_steps):
                #get feed_dict
                feed_dict = ffun_data.fill_feed_dict(train_bh.next_batch(Train_cfg.batch_size),\
                images_placeholder, labels_placeholder, keep_prop_placeholder, mode='train')
                # run train_op and loss op
                _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)
                duration = time.time() - start_time
                #print the message
                if step % 100 == 0:
                    print 'Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration)
                #Save a checkpoint and evaluate the model periodically.
                if (step + 1) % 6000 == 0 or (step + 1) == max_steps:
                    checkpoint_file = os.path.join(Train_cfg.model_dir, 'ffunNet_model.ckpt')
                    saver.save(sess, checkpoint_file, global_step=step)
    pass

if __name__ == '__main__':
    run_train(50000)
