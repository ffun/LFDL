#!/usr/bin/python
# -*- coding: UTF-8 -*-
import tensorflow as tf
import os.path
import time
from ffun.util import *
import ffun_net
import ffun_data

model_dir = '/Users/fang/workspaces/tf_space/model'

def run_train(max_steps):
    data_sets = ffun_data.batch_data()
    data_sets.shuffle(5)#乱序5次
    with tf.Graph().as_default():
        images_placeholder, labels_placeholder, keep_prop_placeholder = ffun_data.placeholder_inputs(50)
        inference = ffun_net.infer(images_placeholder, keep_prop_placeholder)
        loss = ffun_net.loss(inference, labels_placeholder)
        train_op = ffun_net.train(loss, 1e-3)
        init_op = tf.global_variables_initializer()
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(init_op)
            start_time = time.time()
            for step in xrange(max_steps):
                feed_dict = ffun_data.fill_feed_dict(data_sets.next_batch(50), 
                images_placeholder, labels_placeholder, keep_prop_placeholder, mode='train')
                _, loss_value = sess.run([train_op,loss], feed_dict=feed_dict)
                duration = time.time() - start_time
                #print the message
                if step % 100 == 0:
                    print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
                #Save a checkpoint and evaluate the model periodically.
                if (step + 1) % 6000 == 0 or (step + 1) == max_steps:
                    checkpoint_file = os.path.join(model_dir, 'ffunNet_model.ckpt')
                    saver.save(sess, checkpoint_file, global_step=step)
    pass

if __name__ == '__main__':
    run_train(50000)
