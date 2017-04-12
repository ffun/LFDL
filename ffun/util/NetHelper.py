#!/usr/bin/python
# -*- coding: UTF-8 -*-

'''
author:fang.junpeng\n
email:tfzsll@126.com
time:2017-04-10
'''

class Layer(object):
    '''
    所有层的基类,未实现padding
    '''
    def __init__(self, shape, strides=None, padding=None, name=None):
        self.shape = shape
        self.padding = padding
        self.strides = strides
        self.name = self.__class__.__name__
        if name != None:
            self.name = name
    def get_shape(self):
        '得到尺寸'
        return self.shape
    def get_name(self):
        '得到名称'
        return self.name
    def set_name(self, name):
        '设置名称'
        self.name = name
    def get_strides(self):
        '获得步长'
        return self.strides
    def get_padding(self):
        '获得填补数量'
        return self.padding
    def __str__(self):
        '得到吱声的'
        info = self.name+',shape:'+str(self.shape)
        return info
    def out_shape(self, in_shape):
        '''
        根据上一层的输出shape得到当前层输出的shape\n
        Layer默认输出self.shape,子类除了Data_Layer以外,其余的都要重写此方法
        '''
        return self.shape
class Conv_Layer(Layer):
    '卷积层'
    def __init__(self, shape, strides=[1, 1, 1, 1], padding=None, name=None):
        super(Conv_Layer, self).__init__(shape, strides=strides, padding=None, name=None)

    def out_shape(self,in_shape):
        assert len(in_shape) == 3
        Kh0, Kw0, C0 = in_shape
        Kh1, Kw1, C1, num = self.shape
        _, Sh, Sw, _= self.strides
        assert C0 == C1
        H = (Kh0 - Kh1)//Sh + 1
        W = (Kw0 - Kw1)//Sw + 1
        C = num
        return [H, W, C]
    def __str__(self):
        info = super(Conv_Layer, self).__str__()
        info += ',stride:'+str(self.strides)
        return info
    def memory_cost(self):
        '参数内存消耗'
        weight = self.shape[:]
        # weight[-1]是输出feature map个数，加1个bias单元，是总共的神经元个数
        weight[-1] += 1
        return reduce(lambda x, y: x*y, weight)

class Data_Layer(Layer):
    '数据层'
    def out_shape(self, in_shape):
        return self.shape

class Pool_Layer(Layer):
    '池化层'
    def __init__(self, shape, strides=[1, 1, 1, 1], style='max', padding=None, name=None):
        super(Pool_Layer, self).__init__(shape, strides=strides, padding=None, name=None)
        self.style = style
    def __str__(self):
        info = super(Pool_Layer, self).__str__()
        info += ',stride:'+str(self.strides)
        return info
    def out_shape(self, in_shape):
        assert len(in_shape) == 3
        Kh0, Kw0, C0 = in_shape
        _, Kh1, Kw1, _ = self.shape
        _, Sh, Sw, _ = self.strides
        H = (Kh0 - Kh1)//Sh + 1
        W = (Kw0 - Kw1)//Sw + 1
        return [H, W, C0]
    def get_style(self):
        return self.style
class Fc_Layer(Layer):
    '''
    全连接层，层维度只有2维，指明输入和输出的神经元个数即可\n
    example:shape = [128,512]
    '''
    def __init__(self, shape, name=None):
        if len(shape) != 2:
            raise TypeError(Fc_Layer.__name__+' shape should be [in,out]')
        super(Fc_Layer, self).__init__(shape, name=name)
    def out_shape(self, in_shape):
        neurons_in = reduce(lambda x, y: x*y, in_shape)
        n_in, n_out = self.shape
        assert neurons_in == n_in
        return [n_out]
    def memory_cost(self):
        '参数内存消耗'
        weight = self.shape[:]
        # weight[-1]是输出神经元个数，加1个bias单元，是总共的神经元个数
        weight[-1] += 1
        return reduce(lambda x, y: x*y, weight)
class Deconv_Layer(Layer):
    '反卷积层'
    pass

class Net(object):
    '网络'
    def __init__(self, *layers, **param):
        self.__layers = []
        self.name = self.__class__.__name__
        self.__len = 0
        self.shapes = []
        self.name = 'Net'
        if 'name' in param:
            self.name = param['name']
        #添加层，推荐以这种方式
        for layer in layers:
            self.add_layer(layer)
    def add_layer(self, layer):
        '添加层'
        if not isinstance(layer, Layer):
            #判断layer是否为Layer的子类
            raise TypeError("This isn't a Layer")
        self.__layers.append(layer)
        shapes = self.shapes
        if self.__len == 0:
            #直接添加数据层shape
            shapes.append(layer.get_shape())
        else:
            #计算输出的shape
            shape = layer.out_shape(shapes[-1])
            #添加shape
            shapes.append(shape)
        self.__len += 1
        return self
    def layer_num(self):
        '层数'
        if self.__len > 1:
            return self.__len - 1
        else:
            return 0
    def layers(self):
        '返回Net所持有的Layer的副本'
        return self.__layers[:]
    def info(self):
        '打印Net信息'
        print self.name+':'
        layers = self.__layers
        info = ''
        for i in xrange(self.__len):
            info += str(i)+'.'+str(layers[i])
            info += '\n'+'output:'+str(self.shapes[i])+'\n'
        return info
    def weight_memery_cost(self, batch_size=1):
        '计算权重消耗的参数'
        cost = 0
        for layer in self.__layers:
            #检查layer是否具有memory_cost接口
            if hasattr(layer, 'memory_cost'):
                cost += layer.memory_cost()
        return cost*batch_size
    def hidden_memory_cost(self, batch_size=1):
        '计算隐藏层消耗的参数数量'
        cost = 0
        for shape in self.shapes:
            cost += reduce(lambda x, y: x*y, shape)
        cost -= self.data_memory_cost()
        return cost*batch_size
    def data_memory_cost(self, batch_size=1):
        '计算数据层消耗的参数数量'
        cost = 0
        if len(self.shapes) > 0:
            cost = reduce(lambda x, y: x*y, self.shapes[0])
        return cost*batch_size
    def all_memory_cost(self, batch_size=1):
        '计算Net所消耗的参数数量'
        cost = self.weight_memery_cost() + self.hidden_memory_cost() +self.data_memory_cost()
        return cost*batch_size

class NetBuilder(object):
    'for build Net'
    def __init__(self):
        self.NET = None
        pass
    def load_net(self, net):
        'Load '
        assert isinstance(net, Net)
        assert net.layer_num != 0
        self.NET = net
    def create(self):
        'creat net'
        pass
    def conv2d(self, x, w_shape, strides, padding, name, initializer_w=None, initializer_b=None):
        'convolution layer'
        pass
    def max_pool(self, x, ksize, strides, padding, name):
        'max pooling layer'
        pass
    def fc(self, x, w_shape, name, relu=True, initializer_w=None, initializer_b=None):
        'fully connected layer'
        pass
    def dropout(self, x, keep_prop):
        'keep_prpo layer'
        pass
