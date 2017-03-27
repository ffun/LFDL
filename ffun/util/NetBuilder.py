import tensorflow as tf

class Layer(object):
    @classmethod
    def weight_variable(cls, shape, name=None):
        initial = tf.truncated_normal(shape, stddev=0.1, name=name)
        return tf.Variable(initial)
    @classmethod
    def bias_variable(cls, shape, name=None):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial, name=name)
    @classmethod
    def conv(cls, x, W, strides=[1, 1, 1, 1], padding='VALID', name=None):
        return tf.nn.conv2d(x, W, strides=strides, padding=padding, name=name)
    @classmethod
    def pool(cls, x, ksize, strides, padding='VALID', style="max", name=None):
        if cmp(style,"max") == 0:
            return tf.nn.max_pool(x, ksize=ksize, strides=strides, padding=padding, name=name)


