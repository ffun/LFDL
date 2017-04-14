# LFDL  
光场阵列相机深度学习.

## Requirements  
- [TensorFlow1.0][TensorFlow]  
- Python2.7  
- Numpy  
- PIL(for data generation in EPI EPIcreator)
- [**lightfield-analysis/python-tools**][Python Tool](数据集官方提供的工具)

## ffun  
依赖于PIL,numpy,tensorflow等而写的一个python package。  
相关组件(持续更新中)：  
- EPIcreator  
根据数据样本产生原始EPI文件  
- EPIextractor  
输入原始EPI文件，提取用于训练特定大小的EPI文件  
- BatchHelper  
支持乱序、next_batch()方法支持循环取数据  
- NetHelper  
用于研究网络层级的输出关系，会自动判断参数是否正确，以及计算出每一层的输出shape

详情请参照`ffun/doc`目录下的说明文档

## ffun-net使用说明  
1. 下载本代码  
```bash
git clone https://github.com/ffun/LFDL.git
```  
2. 下载光场数据集、解压  
打开`CFG.py`文件，可以看到如下的配置文件:

```python
# file params
Label_DIR = '/Users/fang/workspaces/tf_space/LFDL/disp.txt'#标签位置
Image_DIR = '/Users/fang/workspaces/tf_space/box'#图片位置
EPI_DIR = '/Users/fang/workspaces/tf_space/box/epi36_44'#EPI图像位置
Model_DIR = '/Users/fang/workspaces/tf_space/model'#模型位置

#EPI file
EPI_H = 9
EPI_W = 512
EPI_C = 3
# data params
Data_ALL_NUM = 512*512#总数据量
Data_TRAIN_NUM = 200000#训练集数量
Data_VERIFY_NUM = 40000#验证集数量
Data_TEST_NUM = 22000#测试集数量
# input size of EPT Patch
Input_H = 9#图像高度
Input_W = 33#图像宽度
Input_C = 3#图像通道数
# train params
KEEP_PROP = 0.5# dropout率
LR = 1e-4 # 学习率
Batch_SIZE = 50 #batch-size
Iter_SIZE = (Data_TRAIN_NUM//Batch_SIZE)#[batch-size]个训练数据forward+backward后更新参数过程
Epoch_SIZE = 50#一次epoch=所有训练数据forward+backward后更新参数的过程
```

- 请修改`Image_DIR`的值为box解压的绝对路径，然后运行`python ffunData.py --epi`产生原始epi文件。接着，把原始epi文件所在目录的绝对路径填入`EPI_DIR`的值  

- 修改`Model_DIR`的值，它表示训练好的模型要存放在哪个地方。你还可以修改迭代的次数，学习率等等.  

- 生成disp.txt文件(可选)。如有需要，复制`generate_disp.py`脚本到`/python-tool`目录下，修改其中的训练数据目录、disp存放目录2个参数，然后执行`python generate_disp.py`

3. 一切就绪  
运行`python ffunTrain.py`开始训练网络。大致会显示如下信息：

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

可以看到Loss一直在减小，由于是在Mac上使用i5 CPU处理，所以速度比较慢。  

4. ffun-net网络  
    1. 请打开`ffunNet.py`文件，可以看到结构为"conv,pooling,conv,pooling,fc1,dropout,fc2"，并用softmaxLoss训练网络。整个Net使用面向对象的方法封装成一个整体，并且把执行过程(`sess.run()`)也封装在里面。
    2. conv卷积均为3x3的核，pooling均为1x2的核，在fc1层有1024个神经元，fc1后有个一个dropout层防止过拟合，fc2被分成58个类。

5. 修改与扩展  
    1. 注意：当你修改`CFG.py`中的配置文件，包括`Input_H`、`Input_W`等时，这会改变输入数据的尺寸。此时，你可能需要重新设计卷积尺寸，以及计算出卷积后全连接时的神经元个数，并打开`ffunNet.py`文件修改相关的参数。  
    2. 如果要自定义网络结构，则修改`ffunNet.py`中`infer()`、`loss()`、`eval()`的实现即可(函数名不需要修改)，还可以修改`train()`中的`optimizer`不同的参数更新的方法  
    3. `ffunTrain.py`文件内有一个`run_train()`函数，该函数的作用类似于[***Caffe Framework***][caffe-link]中`Sovler`，主要在其中控制迭代次数、保存模型、测试等。
    4. 在`ffunData.py`中，我封装了一个DataSource，功能很强大，很好用，支持分步加载数据。
    5. 其他的扩展和修改请深入学习tensorflow  

6. 本代码结构参照一个国外小哥fine-tune AlexNet的例子而写，根据面向对象的代码结构，觉得这样代码比较清晰。这个小哥的[**BLOG**][Blog],[**CODE**][CODE]

7. 本代码基础框架还有许多要修改的地方，比如集成模型加载、可视化、增强NetHelper的能力等。我将在后续继续完善。本代码还有比较简陋的r0.1版本，请查看其分支。

[TensorFlow]:https://github.com/tensorflow/tensorflow
[caffe-link]:http://caffe.berkeleyvision.org/
[Blog]:http://kratzert.github.io/2017/02/24/finetuning-alexnet-with-tensorflow.html?utm_source=tuicool&utm_medium=referral
[CODE]:https://github.com/kratzert/finetune_alexnet_with_tensorflow
[Python Tool]:https://github.com/lightfield-analysis/python-tools.git
