#!/usr/bin/python
# -*- coding: UTF-8 -*-
import tensorflow as tf
from ffun.util import *

channels = 3
width = 33
height = 9
#data layer
x = tf.placeholder('float', shape=[None, height*width*channels])
y_ = tf.placeholder('float',shape=[None, 1])

x_image = tf.reshape(x, shape=[-1, height, width, channels])
#conv1 para
w_conv1 = Layer.weight_variable([3, 3, 3, 64], Name="w_conv1")
b_conv1 = Layer.bias_variable([32], Name="b_conv1")

#hidden1
h_conv1 = tf.nn.relu(Layer.conv(x_image, w_conv1, [1, 1, 1, 1]) + b_conv1)
h_pool1 = Layer.pool(h_conv1, Ksize=[1, 1, 2, 1], Strides=[1, 1, 2, 1])

#conv2 para
w_conv2 = Layer.weight_variable([3, 3, 64, 128], Name="w_conv2")
b_conv2 = Layer.bias_variable([64], Name="b_conv2")

#hidden2
h_conv2 = tf.nn.relu(Layer.conv(h_pool1, w_conv2, [1, 1, 1, 1]) + b_conv2)
h_pool2 = Layer.pool(h_conv2, Ksize=[1, 1, 2, 3], Strides=[1, 1, 2, 1])

#w_fc1
w_fc1 = Layer.weight_variable([4*6*64,128])
b_fc1 = Layer.weight_variable([128])

h_pool2_flat = tf.reshape(h_pool2, [-1, 4*4*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)

#w_fc2
w_fc2 = Layer.weight_variable([128,1])
b_fc2 = Layer.weight_variable([1])

y = tf.matmul(h_fc1, w_fc2) + b_fc2


loss = tf.reduce_mean(tf.square(y-y_))
optimizer = tf.train.GradientDescentOptimizer(1e-1)

train = optimizer.minimize(loss)








