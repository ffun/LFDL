# NetHelper 
- NetHelper用于研究网络层级的输出关系，会自动判断参数是否正确，以及计算出每一层的输出shape。  
- NetHelper是独立于任何框架的Net描述中间件，随着ffun package的演进，以后也许可以使用NetHelper自动产生tensorflow、Mxnet等框架的代码
```python
#!/usr/bin/python
# -*- coding: UTF-8 -*-
#导入包
import NetHelper
#使用方式1
#创建Layer对象
L1 = NetHelper.Data_Layer([9, 33, 3])
L2 = NetHelper.Conv_Layer([3, 3, 3, 64])
L3 = NetHelper.Pool_Layer([1, 1, 2, 1], [1, 1, 2, 1])
L4 = NetHelper.Conv_Layer([3, 3, 64, 128])
L5 = NetHelper.Pool_Layer([1, 1, 2, 1], [1, 1, 2, 1])
L6 = NetHelper.Fc_Layer([128*5*6, 1024])
L7 = NetHelper.Fc_Layer([1024, 58])
#创建Net并添加Layer,其中Data_Layer必须被第一个添加
net = NetHelper.Net(L1, L2, L3, L4, L5, L6, L7)#Net.add_layer() API添加层
#方式2
net = NetHelper.Net(
    NetHelper.Data_Layer([9, 33, 3]),
    NetHelper.Conv_Layer([3, 3, 3, 64]),
    NetHelper.Pool_Layer([1, 1, 2, 1], [1, 1, 2, 1]),
    NetHelper.Conv_Layer([3, 3, 64, 128]),
    NetHelper.Pool_Layer([1, 1, 2, 1], [1, 1, 2, 1]),
    NetHelper.Fc_Layer([128*5*6, 1024]),
    NetHelper.Fc_Layer([1024, 58])
)
#打印网络信息
print net.info()
#打印网络层数，数据层不算
print 'layer num:', net.layer_num()
#输出内存消耗(单位为参数个数)。需要根据每个点的数据类型(比如int,float)重新换算
#以下4个函数，默认batch-size=1情况下的消耗，可以传入一个参数表示batch-size
print 'weight_memery_cost:', net.weight_memery_cost()
print 'hidden_memory_cost:', net.hidden_memory_cost()
print 'data_memory_cost:', net.data_memory_cost()
print 'all_memort_cost:', net.all_memory_cost()
```

输出的信息类似如下：

```bash
Net:
0.Data_Layer,shape:[9, 33, 3]
output:[9, 33, 3]
1.Conv_Layer,shape:[3, 3, 3, 64],stride:[1, 1, 1, 1]
output:[7, 31, 64]
2.Pool_Layer,shape:[1, 1, 2, 1],stride:[1, 1, 2, 1]
output:[7, 15, 64]
3.Conv_Layer,shape:[3, 3, 64, 128],stride:[1, 1, 1, 1]
output:[5, 13, 128]
4.Pool_Layer,shape:[1, 1, 2, 1],stride:[1, 1, 2, 1]
output:[5, 6, 128]
5.Fc_Layer,shape:[3840, 1024]
output:[1024]
6.Fc_Layer,shape:[1024, 58]
output:[58]

layer num: 6
weight_memery_cost: 4072475
hidden_memory_cost: 33850
data_memory_cost: 891
all_memort_cost: 4107216
```