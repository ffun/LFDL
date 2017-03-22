import tensorflow as tf

class Layer(object):
    @classmethod
    def weight_variable(cls, shape, Name=None):
        initial = tf.truncated_normal(shape, stddev=0.1, name=Name)
        return tf.Variable(initial)
    @classmethod
    def bias_variable(cls, shape, Name=None):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial, name=Name)
    @classmethod
    def conv(cls, x, W, Strides=[1, 1, 1, 1], Padding='SAME', Name=None):
        return tf.nn.conv2d(x, W, strides=Strides, padding=Padding, name=Name)
    @classmethod
    def pool(cls, x, Ksize, Strides, Padding='SAME', style="max", Name=None):
        if cmp(style,"max") == 0:
            return tf.nn.max_pool(x, ksize=Ksize, strides=Strides, padding=Padding, name=Name)


