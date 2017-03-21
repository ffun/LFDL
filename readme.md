# LFDL  
光场阵列相机深度学习.

# ffun
自己写的一个python package,依赖于PIL,numpy,scipy,tensorflow等  
## ffun Instruction  
- 导入package  
```python
import ffun.io as Fio
import ffun.util as Fut
```
- EPI Genration  
```python
#得到文件列表，入参为文件目录。
#返回的文件列表是经过排序的(排序规则为字典序)
files =  Fio.FileHelper.get_files('/Users/fang/workspaces/tf_space/LFDL/pngdata')
#创建EPI生成器对象，入参为文件列表的元组
Epi_creator = Fio.EPIcreator(files)
#生成EPI文件，入参为图片索引闭区间
Epi_creator.create((36,44))
```  
上述代码会在`pngdata`目录下产生`epi36_44`目录，并在目录下产生epi36_44_000~511.png（假设原始图像height = 512）。生成的epi数据是原始epi，实际训练可能只是其中的一个窗口数据，此时需要调用`EPIextractor`提供的方法。  

- EPI extract  
```python
#得到提取器对象，该对象会持有图片的Numpy形式存储，因此只要加载一次文件就可以反复提取
extractor = Fio.EPIextractor('/Users/fang/workspaces/tf_space/LFDL/pngdata/epi36_44/epi_36_44_001.png')
#提取x坐标点为100处，默认长度为32的窗口，高度为图像高度
extractor.extract(100)
```
- Label get  
```python
#得到标签加载器对象
labelloader = Fio.TextLoader()
#得到标签，tuple元组类型。
#入参为标签文件和元素转换函数，此处将数据转换为float。
label = labelloader.read('/Users/fang/workspaces/tf_space/LFDL/disp.txt',float)
```  
- batch乱序  
batch-data在输入网络时最好是先经过乱序，这样训练处的模型鲁棒性会更强。
```python  
#数据batch和label序列，需要保证长度一致，可以是list或者tuple类型
batch = [1,2,3,4,5]
label = [1,2,3,4,5]
#实例化BatchHelper
bh = Fut.BatchHelper((batch,label))
#乱序
bh.shuffle()
#拿到队头元素，注意到BatchHelper.head()方法返回的是tuple类型，其顺序和构造函数的入参一致
current_data = bh.head()[0]
current_label = bh.head()[1]
print current_data,current_label
```

