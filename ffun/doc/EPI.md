# ffun.EPI package  
此包主要处理一些io相关的操作
- 导入package  
```python
from ffun.EPI import*
from ffun.FileHelper import*
```  
## EPI  
EPI生成器

```python
示例:
#得到文件列表，入参为文件目录，文件后缀。返回经过排序的文件名
files =  FileHelper.get_files('/Users/fang/workspaces/tf_space/box', '.png')
#创建EPI生成器对象，入参为文件列表的元组
epi = EPI(files)
#生成EPI文件，入参分别是图片索引序列(可以是任意的索引序列)、epi方向(u--水平，v--竖直方向)、存放路径
epi.create(range(36, 45), 'u','/Users/fang/workspaces/tf_space/test/EPI-u')
```  
上述代码会在`EPI-u`目录下产生`epi45_53`目录，并在目录下产生000~511.png（假设原始图像height = 512)。EPI在creat()的时候，会根据水平或竖直方向、以及图片的长宽，自动计算EPI的尺寸和通道数。  

## PatchHelper  
Patch提取助手

```python
#图片地址，这张图片的shape是(9, 512, 3)
path = '/Users/fang/workspaces/tf_space/test/EPI-u/000.png'
#读取图片
ih = ImageHelper().read(path)
#创建PathHelper对象，入参是图片的numpy.ndarray类型数据
ph = PatchHelper(ih.data_convert3d())
#设置padding(可选)，入参是4维向量，分别是上、下、左、右的pad数量
ph.padding([0, 0, 16, 16])
#提取数据，入参是卷积核，步长。如果图片是(512, 9, 3)(竖着的EPI)
#那么卷积核就应该为[33, 9]。根据图片选择。卷积核大小是任意的，但是超过图片尺寸的话，提取到的数量会为0.请自己计算
ph.extract([9, 33], [1, 1])
#拿到提取到的patches,这是一个list
patch_list = ph.patches()
print ph.size()
```

