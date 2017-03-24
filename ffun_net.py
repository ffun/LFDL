#!/usr/bin/python
# -*- coding: UTF-8 -*-
import tensorflow as tf
from ffun.util import *
import ffun_data
'''
channels = 3
width = 33
height = 9
'''
channels = ffun_data.img_cfg['channel']
width = ffun_data.img_cfg['width']
height = ffun_data.img_cfg['height']

def infer(images):
    '''
    function:infer function is to build net,give the forward compute\n
    @images:Images placeholder from ffun_data
    '''
    #conv1 para
    w_conv1 = Layer.weight_variable([3, 3, 3, 64], Name="w_conv1")
    b_conv1 = Layer.bias_variable([32], Name="b_conv1")

    #hidden1
    h_conv1 = tf.nn.relu(Layer.conv(images, w_conv1, [1, 1, 1, 1]) + b_conv1)
    h_pool1 = Layer.pool(h_conv1, Ksize=[1, 1, 2, 1], Strides=[1, 1, 2, 1])

    #conv2 para
    w_conv2 = Layer.weight_variable([3, 3, 64, 128], Name="w_conv2")
    b_conv2 = Layer.bias_variable([64], Name="b_conv2")

    #hidden2
    h_conv2 = tf.nn.relu(Layer.conv(h_pool1, w_conv2, [1, 1, 1, 1]) + b_conv2)
    h_pool2 = Layer.pool(h_conv2, Ksize=[1, 1, 2, 1], Strides=[1, 1, 2, 1])

    #w_fc1
    w_fc1 = Layer.weight_variable([5*6*128, 1024])
    b_fc1 = Layer.weight_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 5*6*128])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)

    #dropout
    keep_prob = tf.placeholder('float')
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    #w_fc2
    w_fc2 = Layer.weight_variable([1024, 1])
    b_fc2 = Layer.weight_variable([1])
    #output
    inference = tf.matmul(h_fc1_drop, w_fc2) + b_fc2

    return inference

def loss(inference, label):
    '''
    function:loss function use the inference and label to compute loss
    '''
    #回归的损失函数
    return tf.reduce_mean(tf.square(inference - label))

def train(loss, lr):
    '''
    function to train the net\n
    @loss:loss tensor from loss()\n
    @lr:The learning rate to use for gradient descent
    '''
    # Create the gradient descent optimizer with the given learning rate.
    optimizer = tf.train.GradientDescentOptimizer(lr)
    # Create a variable to track the global step
    global_step = tf.Variable(0, name='global_step', trainable=False)
    # Use the optimizer to apply the gradients that minimize the loss
    # (and also increment the global step counter) as a single training step.
    train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op

def evaluation(inference, labels):
    '''
    function to evaluation the DL model\n
    @inference:inference tensor from infer()\n
    @labels:labels tensor from ffun_data
    '''
    pass
