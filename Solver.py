#!/usr/bin/python
# -*- coding: UTF-8 -*-
import tensorflow as tf
from ffunNet import ffunNet
import CFG
import ffun_data
import time
import os.path
from ffun.DataProvider import*

class Logger(object):
    @staticmethod
    def log(info, logpath):
        with open(logpath, 'a') as f:
            f.write(info)

def run_train(max_steps):
    '''
    run train
    '''
    #get data:训练集，验证集合测试集
    tr_bh, vf_bh, te_bh = ffun_data.data()
    #get batch_size
    bs = CFG.Batch_SIZE
    #get DataSet
    train_bh, verify_bh, test_bh = DataSet(tr_bh, bs), DataSet(vf_bh, bs), DataSet(te_bh, bs)
    # Net
    net = ffunNet(CFG.LR)
    #graph
    with tf.Graph().as_default():
        #get placeholder
        images_pl, labels_pl, prop_pl = ffun_data.placeholder_inputs(CFG.Batch_SIZE)
        #构建网络
        net.build(images_pl, labels_pl, prop_pl)
        train_op = net.train()#train_op一定要在初始化前在计算图中明确定义出，否则会出现未初始化错误
        #saver
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())# init variables
            start_time = time.time()
            for i in xrange(max_steps):
                for step in xrange(CFG.Iter_SIZE):
                    #loss_value = net.run_train(sess, train_bh)
                    feed_dict = ffun_data.fill_feed_dict(train_bh.next_batch(),\
                    images_pl, labels_pl, prop_pl, mode='train')
                    # run train_op and loss op
                    #_, loss_value = sess.run([train_op,net.loss()], feed_dict=feed_dict)
                    loss_value = net.run_train(sess, train_bh)
                    duration = time.time() - start_time
                    #print the message
                    if step % 500 == 0:
                        tran_info = 'Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration)
                        print tran_info
                        Logger.log(tran_info+'\n', CFG.Model_DIR+'/log.txt')#logging
                        
                    #Save a checkpoint and evaluate the model periodically.
                if (i+1) % 10 == 0 or (i+1) == max_steps:#每10w次评估下
                    #eval
                    eval_info = 'Step '+str(i)+':'
                    eval_info += 'Validation Data Eval:'
                    print eval_info
                    Logger.log(eval_info+'\n', CFG.Model_DIR+'/log.txt')
                    eval_info = net.run_eval(sess, verify_bh)
                    print eval_info
                    Logger.log(eval_info+'\n', CFG.Model_DIR+'/log.txt')
                    #test feed
                    test_info = 'Step '+str(i)+':'
                    test_info += 'Test Data Eval:'
                    Logger.log(test_info+'\n', CFG.Model_DIR+'/log.txt')
                    print test_info
                    test_info = net.run_eval(sess, test_bh)
                    Logger.log(test_info+'\n', CFG.Model_DIR+'/log.txt')
                    print test_info
                    #keep model
                    checkpoint_file = os.path.join(CFG.Model_DIR, 'ffun-net.ckpt')
                    saver.save(sess, checkpoint_file, global_step=step)
    return

if __name__ == '__main__':
    run_train(CFG.Epoch_SIZE)#训练epoch-size个全部数据的forward和backword