# ffun.util package  
此包主要是一些方便的工具，将深度学习中经常要用到的操作进行封装。
- 导入package  
```python
import ffun.util as Fut
```  
## BatchHelper
- batch乱序  
batch-data在输入网络时最好是先经过乱序，这样训练处的模型鲁棒性会更强。

```python  
#数据batch和label序列，需要保证长度一致，可以是list或者tuple类型
batch = [1,2,3,4,5,6,7]
label = [1,2,3,4,5,6,7]
#实例化BatchHelper
bh = Fut.BatchHelper((batch,label))
#乱序,可传入乱序次数，默认乱序1次
bh.shuffle()
#1.拿到队头元素，注意到BatchHelper.head()方法返回的是tuple类型，其顺序和构造函数的入参一致
current = bh.head()
#judge if current is None
if current != None:
    d = current[0]
    l = current[1]
    print d,l
#2.拿到一个batch_size的数据.建议batch_size值小于bh所持有的队列长度.ps:该方法支持循环得到batch
current = bh.next_batch(5)
datas = current[0]#len(data)=5
labels = current[1]#len(label)=5
```  

**提醒**：以上是对加载到内存中的2个序列创建BatchHelper对象，然后进行乱序。然而，当数据足够大时并不能全部加载到内存后再进行乱序。此时，可以生成数据路径的乱序索引文件，然后在训练时对该文件进行加载。  

## NetCalculator  
该工具是用于研究层级的输出关系，会自动判断参数是否正确，以及计算出每一层的输出shape，提升效率。

```python
#导入包
import ffun.util as Fut
#创建对象
nc = Fut.NetCalculator()
#第一步是设置数据层，否则会报错
nc.set_dataLayer([9,33,3])
#添加层信息
nc.add_conv_layer(ksize=[3, 3, 3, 64])
nc.add_pool_layer(ksize=[1,1,2,1],strides=[1,1,2,1])
nc.add_conv_layer(ksize=[3, 3, 64, 128])
nc.add_pool_layer(ksize=[1,1,2,1],strides=[1,1,2,1])
nc.add_fc_layer([128*5*6,1024])
#print messge
nc.print_layers()
print 'layer num:',nc.num_of_layers()
#输出内存消耗(单位为参数个数)。需要根据每个点的数据类型(比如int,float)重新换算
print 'weight_memery_cost:',nc.weight_memery_cost()
print 'hidden_memory_cost:',nc.hidden_memory_cost()
print 'data_memory_cost:',nc.data_memory_cost()
print 'all_memort_cost:',nc.all_memort_cost()
```

输出的信息类似如下（output行在输出时会显示为绿色）：

```bash
input:[9, 33, 3]
conv1--ksize:[3, 3, 3, 64];strides:[1, 1, 1, 1]
output:[7, 31, 64]
pool1--ksize:[1, 1, 2, 1];strides:[1, 1, 2, 1]
output:[7, 15, 64]
conv2--ksize:[3, 3, 64, 128];strides:[1, 1, 1, 1]
output:[5, 13, 128]
pool2--ksize:[1, 1, 2, 1];strides:[1, 1, 2, 1]
output:[5, 6, 128]
weight_memery_cost: 4012059
hidden_memory_cost: 33792
data_memory_cost: 891
all_memort_cost: 4046742
```
