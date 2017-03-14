# LFDL  
光场阵列相机深度学习.

# ffun
自己写的一个python package,依赖于PIL,numpy,scipy,tensorflow等  
## Simple Instruction  
- EPI Genration  
```python
import ffun.io as Fio
#得到文件列表，入参为文件目录。
#返回的文件列表是经过排序的，排序规则为字典序
files =  Fio.FileHelper.get_files('/Users/fang/workspaces/tf_space/LFDL/pngdata')
#1.创建EPI生成器对象，入参为文件列表的元组
#2.成功创建对象后，该对象会持有目录下png图像的有序元组
Epi_creator = Fio.EPIcreator(files)
#生成EPI文件，入参为图片索引闭区间
Epi_creator.create((36,44))
```  
上述代码会在`pngdata`目录下产生`epi36_44`目录，并在目录下产生epi36_44_000~511.png（假设原始图像height = 512）。生成的epi数据是原始epi，实际训练可能只是其中的一个窗口数据，此时需要调用`EPIextractor`提供的方法。  

- EPI extract  
```python
extractor = Fio.EPIextractor('/Users/fang/workspaces/tf_space/LFDL/pngdata/epi36_44/epi_36_44_001.png')
extractor.extract(100)
```
- Label get  
```python
#得到标签加载器对象
labelloader = Fio.TextLoader()
#得到标签，tuple元组类型
label = labelloader.read('/Users/fang/workspaces/tf_space/LFDL/disp.txt')
```
