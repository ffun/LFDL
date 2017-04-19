# ffun.EPI package  
此包主要处理一些io相关的操作
- 导入package  
```python
from ffun.EPI import*
from ffun.FileHelper import*
```  
## EPI
```python
示例:
#得到文件列表，入参为文件目录，文件后缀。返回经过排序的文件名
files =  FileHelper.get_files('/Users/fang/workspaces/tf_space/box', '.png')
#创建EPI生成器对象，入参为文件列表的元组
epi = EPI(files)
#生成EPI文件，入参分别是图片索引序列(可以是任意的索引序列)、epi方向(u--水平，v--竖直方向)
epi.create(range(36, 45), 'u','/Users/fang/workspaces/tf_space/test/EPI-u')
```  
上述代码会在`EPI-u`目录下产生`epi45_53`目录，并在目录下产生000~511.png（假设原始图像height = 512)。EPI在creat()的时候，会根据水平或竖直方向、以及图片的长宽，自动计算EPI的尺寸和通道数。  

## EPI extract  
```python
#得到提取器对象，该对象会持有图片的Numpy形式存储，因此只要加载一次文件就可以反复提取
extractor = EPIextractor('/Users/fang/workspaces/tf_space/LFDL/pngdata/epi36_44/epi_36_44_001.png')
#提取x坐标点为100处，默认长度为32的窗口，高度为图像高度
extractor.extract(100)
```
## Label get  
```python
#得到标签加载器对象
labelloader = LabelHelper()
#得到标签，tuple元组类型。
#入参为标签文件和元素转换函数，此处将数据转换为float。
label = labelloader.read('/Users/fang/workspaces/tf_space/LFDL/disp.txt',float)
```  