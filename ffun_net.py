#!/usr/bin/python
# -*- coding: UTF-8 -*-
import tensorflow as tf
from ffun.util import *
import ffun_data

channels = ffun_data.img_cfg['channel']
width = ffun_data.img_cfg['width']
height = ffun_data.img_cfg['height']

def infer(images, keep_prob):
    '''
    function:infer function is to build net,give the forward compute\n
    @images:Images placeholder from ffun_data\n
    @keep_prob:dropout layer's keep_prop
    '''
    #conv1 para
    w_conv1 = Layer.weight_variable([3, 3, 3, 64], name="w_conv1")
    b_conv1 = Layer.bias_variable([64], name="b_conv1")

    #hidden1
    h_conv1 = tf.nn.relu(Layer.conv(images, w_conv1, [1, 1, 1, 1]) + b_conv1)
    h_pool1 = Layer.pool(h_conv1, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1])

    #conv2 para
    w_conv2 = Layer.weight_variable([3, 3, 64, 128], name="w_conv2")
    b_conv2 = Layer.bias_variable([128], name="b_conv2")

    #hidden2
    h_conv2 = tf.nn.relu(Layer.conv(h_pool1, w_conv2, [1, 1, 1, 1]) + b_conv2)
    h_pool2 = Layer.pool(h_conv2, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1])

    #w_fc1
    w_fc1 = Layer.weight_variable([5*6*128, 1024],stddev=1.0/math.sqrt(5*6*127))
    b_fc1 = Layer.weight_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 5*6*128])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)

    #dropout
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    #w_fc2
    ''' 
    # if use regression
    w_fc2 = Layer.weight_variable([1024, 1])
    b_fc2 = Layer.weight_variable([1])
    #output
    inference = tf.matmul(h_fc1_drop, w_fc2) + b_fc2
    '''
    # if use classify
    w_fc2 = Layer.weight_variable([1024, 58], stddev=1.0/math.sqrt(1024))
    b_fc2 = Layer.weight_variable([58])
    inference = tf.matmul(h_fc1_drop, w_fc2) + b_fc2
    
    return inference

def loss(inference, labels):
    '''
    function:loss function use the inference and label to compute loss
    '''
    '''
    #回归的损失函数
    return tf.reduce_mean(tf.square(inference - labels), name='L2_Loss_mean')
    '''
    #分类的softmaxLoss,采用信息熵形式
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=inference, name='xentropy')
    return tf.reduce_mean(cross_entropy)

def train(loss, lr):
    '''
    function to train the net\n
    @loss:loss tensor from loss()\n
    @lr:The learning rate to use for gradient descent
    '''
    # Create the gradient descent optimizer with the given learning rate.
    #optimizer = tf.train.GradientDescentOptimizer(lr)
    optimizer = tf.train.AdamOptimizer(lr, 0.9, 0.995)
    # Create a variable to track the global step
    global_step = tf.Variable(0, name='global_step', trainable=False)
    # Use the optimizer to apply the gradients that minimize the loss
    # (and also increment the global step counter) as a single training step.
    train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op

def eval(inference, labels):
    '''
    function to evaluation the DL model\n
    @inference:inference tensor from infer()\n
    @labels:labels tensor from ffun_data
    '''
    '''
    diff = inference - labels
    diff = tf.abs(diff)
    return diff
    '''
    correct = tf.nn.in_top_k(inference, labels, 1)
    #correct = tf.equal(tf.argmax(inference, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_sum(tf.cast(correct, tf.int32))
    return accuracy

