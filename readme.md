# LFDL  
光场阵列相机深度学习.

## ffun
自己写的一个python package,依赖于PIL,numpy,scipy,tensorflow等。  
请参照ffun.doc目录下的说明文档

## ffun-net使用说明  
1. 下载本代码  
```bash
git clone https://github.com/ffun/LFDL.git
```  
2. 下载box数据集、解压，修改配置文件  
打开`ffun_data.py`文件，可以看到如下的配置文件。
- 配置文件1

```python
#配置文件
train_data_cfg = {
    'img-dir':'/Users/fang/workspaces/tf_space/box',
    'origin-epi-dir':'/Users/fang/workspaces/tf_space/box/epi45_53',
    'label-dir':'/Users/fang/workspaces/tf_space/LFDL/disp.txt'
}
#img 配置文件
img_cfg = {
    'height':9,
    'width':512,
    'channel':3
}
#epi 配置文件
epi_cfg = {
    'height':9,
    'width':33,#需要提取的epi长度
    'channel':3,
    'mode':'c=0'
}
```

请修改`train_data_cfg['img-dir']`的值为box解压的绝对路径，然后运行`python ffun_data.py`产生原始epi文件。接着，把原始epi文件所在目录的绝对路径填入`train_data_cfg['origin-epi-dir']`的值  

- 配置文件2  
打开`ffun_train.py`文件，修改model_dir的值，它表示训练好的模型要存放在哪个地方。你还可以在文件中修改迭代的次数，学习率等等。  

```python
model_dir = '/Users/fang/workspaces/tf_space/model'
```  

3. 一切就绪  
运行`python ffun_train.py`开始训练网络。大致会显示如下信息：

```bash
Step 1800: loss = 102298310052023873715568640.00 (406.326 sec)
Step 1900: loss = 68546643685122870750478336.00 (428.306 sec)
······
Step 10400: loss = 113548214272.00 (2416.193 sec)
Step 10500: loss = 76084084736.00 (2440.081 sec)
······
Step 15500: loss = 151.48 (3648.776 sec)
Step 15600: loss = 104.22 (3671.850 sec)
······
Step 16400: loss = 5.82 (3870.151 sec)
Step 16500: loss = 3.65 (3895.443 sec)
······
Step 19100: loss = 0.60 (4519.097 sec)
Step 19200: loss = 0.59 (4543.488 sec)
····
```

可以看到Loss一直在减小，由于是在Mac上使用i5 CPU处理，所以速度比较慢(而且随着迭代次数增加而变慢)。  

4. ffun-net网络  
    1. 请打开`ffun_net.py`文件，可以看到结构为"conv,pooling,conv,pooling,fc1,dropout,fc2"，并为L2损失训练网络。
    2. conv卷积均为3x3的核，pooling均为1x2的核，在fc1层有1024个神经元。由于采用回归方式，所以fc2回归到1个神经元，输出一个float型实数。

5. 修改与扩展  
    1. 注意：当你修改`ffun_data.py`中的配置文件，包括`img_cfg`、`epi_cfg`时，这会改变输入数据的尺寸。此时，你可能需要重新设计卷积尺寸，以及计算出卷积后全连接时的神经元个数，并打开`ffun_net.py`文件修改相关的参数。  
    2. 如果要自定义网络结构，则修改`ffun_net.py`中`infer()`、`loss`的实现即可(函数名不需要修改)，还可以修改`train()`中的`optimizer`不同的参数更新的方法(在本代码中使用的是简单梯度下降方式)  
    3. `ffun_train.py`文件内只有一个`run_train()`函数，该函数的作用类似于[***Caffe Framework***][caffe-link]中`Sovler`，主要在其中控制迭代次数、保存模型、测试等。还有就是给网络喂`feed_dict`(tensorflow特有的两种传递参数的方式之一)。
    4. 其他的扩展和修改请深入学习tensorflow  

6. 本代码结构参照tensorflow的mnist例子而写，但是网络模型、训练细节等方面有一些不同。tensorflow mnist代码链接：[**mnist.py**][mnist-code]、[**fully_connected_feed.py**][fully_connected_feed-code]


[caffe-link]:http://caffe.berkeleyvision.org/
[mnist-code]:https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/examples/tutorials/mnist/mnist.py
[fully_connected_feed-code]:https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/examples/tutorials/mnist/fully_connected_feed.py