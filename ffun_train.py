#!/usr/bin/python
# -*- coding: UTF-8 -*-
import tensorflow as tf
import os.path
import time
from ffun.util import *
import ffun_net
import ffun_data

#配置文件
Train_CFG = {
    'model_dir':'/Users/fang/workspaces/tf_space/model',
    'eval_region': 0.07,
    'batch-size':50
}

# 模型评估
def do_eval(sess, eval_correct, images_pl, labels_pl, prop_pl, dataset):
    true_count = 0.0
    step_epochs = dataset.num() // dataset.batch_size()
    for step in xrange(step_epochs):
        #get feed_dict
        feed_dict = ffun_data.fill_feed_dict(dataset.next_batch(), \
        images_pl, labels_pl,  prop_pl, mode='test')
        #run
        '''
        #回归准确率计算
        diff = sess.run(eval_correct, feed_dict=feed_fict)
        true_count += SeqHelper.stat_seq(diff[0], Train_CFG['eval_region'])
        '''
        #分类准确率计算
        true_count += sess.run(eval_correct, feed_fict=feed_dict)
    #准确率
    precision = float(true_count)/dataset.num()

    print 'num_examples:%d,correct:%d,precision:%0.04f' % (dataset.num(), true_count, precision)

def run_train(max_steps):
    '''
    run train
    '''
    #get data:训练集，验证集合测试集
    tr_bh, vf_bh, te_bh = ffun_data.data()
    #get batch_size
    bs = Train_CFG['batch-size']
    #get DataSet
    train_bh, verify_bh, test_bh = DataSet(tr_bh, bs), DataSet(tr_bh, bs), DataSet(te_bh, bs)
    #graph
    with tf.Graph().as_default():
        #get placeholder
        images_pl, labels_pl, prop_pl = ffun_data.placeholder_inputs(train_bh.batch_size())
        #get inference
        inference = ffun_net.infer(images_pl, prop_pl)
        #correct eval
        eval_correct = ffun_net.eval(inference, labels_pl)
        #get loss
        loss = ffun_net.loss(inference, labels_pl)
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
                feed_dict = ffun_data.fill_feed_dict(train_bh.next_batch(),\
                images_pl, labels_pl, prop_pl, mode='train')
                # run train_op and loss op
                _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)
                duration = time.time() - start_time
                #print the message
                if step % 100 == 0:
                    print 'Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration)
                #Save a checkpoint and evaluate the model periodically.
                if (step+1) % 5000 == 0 or (step+1) == max_steps:
                    checkpoint_file = os.path.join(Train_CFG['model_dir'], 'ffunNet_model.ckpt')
                    saver.save(sess, checkpoint_file, global_step=step)
                    #eval
                    print 'Validation Data Eval:'
                    do_eval(sess, eval_correct, images_pl, labels_pl, prop_pl, verify_bh)
                    print 'Test Data Eval:'
                    do_eval(sess, eval_correct, images_pl, labels_pl, prop_pl, test_bh)
    pass

if __name__ == '__main__':
    run_train(30000)
