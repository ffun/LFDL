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
<font color=#FF7F50>提醒：</font>以上是对加载到内存中的2个序列创建BatchHelper对象，然后进行乱序。然而，当数据足够大时并不能全部加载到内存后再进行乱序。此时，可以生成数据路径的乱序索引文件，然后在训练时对该文件进行加载。