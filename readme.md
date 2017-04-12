# LFDL  
光场阵列相机深度学习.

## Requirements  
- Tensorflow1.0  
- Python2.7  
- Numpy  
- PIL(for data generation in EPI EPIcreator)

## ffun  
依赖于PIL,numpy,tensorflow等而写的一个python package。  
相关组件(持续更新中)：  
- EPIcreator  
根据数据样本产生原始EPI文件  
- EPIextractor  
输入原始EPI文件，提取用于训练特定大小的EPI文件  
- BatchHelper、DataSet  
支持乱序、next_batch()方法支持循环取数据  
- NetHelper  
用于研究网络层级的输出关系，会自动判断参数是否正确，以及计算出每一层的输出shape

详情请参照`ffun/doc`目录下的说明文档

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
    'origin-epi-dir':'/Users/fang/workspaces/tf_space/box/epi36_44',
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
#配置文件:训练、验证、测试集的数量
data_cfg = {
    'all':512*512,
    'train':200000,
    'verify':40000,
    'test':22000
}
```

请修改`train_data_cfg['img-dir']`的值为box解压的绝对路径，然后运行`python ffun_data.py --epi`产生原始epi文件。接着，把原始epi文件所在目录的绝对路径填入`train_data_cfg['origin-epi-dir']`的值  

- 配置文件2  
打开`ffun_train.py`文件，修改model_dir的值，它表示训练好的模型要存放在哪个地方。你还可以在文件中修改迭代的次数，学习率等等。  

```python
#配置文件
Train_CFG = {
    'model_dir':'/Users/fang/workspaces/tf_space/model',
    'eval_region': 0.07,
    'batch-size':100
}
```  

3. 一切就绪  
运行`python ffun_train.py`开始训练网络。大致会显示如下信息：

```bash
generate epi done!
load labels done!
generate data done!
Step 0: loss = 5.24 (0.132 sec)
Step 100: loss = 1.44 (11.945 sec)
Step 200: loss = 1.41 (22.117 sec)
Step 300: loss = 1.11 (32.315 sec)
······
```

可以看到Loss一直在减小，由于是在Mac上使用i5 CPU处理，所以速度比较慢(而且随着迭代次数增加而变慢)。  

4. ffun-net网络  
    1. 请打开`ffun_net.py`文件，可以看到结构为"conv,pooling,conv,pooling,fc1,dropout,fc2"，并为L2损失训练网络。
    2. conv卷积均为3x3的核，pooling均为1x2的核，在fc1层有1024个神经元。如果由于采用回归方式，那么fc2回归到1个神经元，输出一个float型实数;如果采用分类的方式，fc2被分成58个类。

5. 修改与扩展  
    1. 注意：当你修改`ffun_data.py`中的配置文件，包括`img_cfg`、`epi_cfg`时，这会改变输入数据的尺寸。此时，你可能需要重新设计卷积尺寸，以及计算出卷积后全连接时的神经元个数，并打开`ffun_net.py`文件修改相关的参数。  
    2. 如果要自定义网络结构，则修改`ffun_net.py`中`infer()`、`loss()`、`eval()`的实现即可(函数名不需要修改)，还可以修改`train()`中的`optimizer`不同的参数更新的方法(在本代码中使用的是简单梯度下降方式)  
    3. `ffun_train.py`文件内有一个`run_train()`函数，该函数的作用类似于[***Caffe Framework***][caffe-link]中`Sovler`，主要在其中控制迭代次数、保存模型、测试等。还有就是给网络喂`feed_dict`(tensorflow特有的两种传递参数的方式之一)。此外还有一个`do_eval()`函数是用于做模型评估的。
    4. 其他的扩展和修改请深入学习tensorflow  

6. 本代码结构参照tensorflow的mnist例子而写，但是网络模型、训练细节等方面有一些不同。tensorflow mnist代码链接：[**mnist.py**][mnist-code]、[**fully_connected_feed.py**][fully_connected_feed-code]


[caffe-link]:http://caffe.berkeleyvision.org/
[mnist-code]:https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/examples/tutorials/mnist/mnist.py
[fully_connected_feed-code]:https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/examples/tutorials/mnist/fully_connected_feed.py