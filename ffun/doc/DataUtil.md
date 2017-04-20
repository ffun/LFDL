# ffun.DataUtil  
此模块主要负责数据预处理  
- 导入package  
```python
from ffun.DataUtil import*
```  

## ImageHelper  
图像的包装类，提供读取、存储、获取图像信息等功能

```python
#新建ImageHelper对象
ih = ImageHelper()
#读入一张图片
ih.read('/Users/fang/workspaces/tf_space/box/input_Cam007.png')
#显示它的shape(numpy.ndarray的shape)
print ih.shape()
#保存图片
ih.save('/Users/fang/workspaces/tf_space/t.png')
#拿到它的数据(numpy.ndarray类型)，可以对此data进行一定处理
data = ih.data()
#转换成3维形式
data = ih.data_convert3d()
```

## ImageCollection  

可以通过这个类得到多个图像叠在一起的数据结构

```python
gray_img = '/Users/fang/workspaces/tf_space/test/t1.png'
color_img = '/Users/fang/workspaces/tf_space/test/t.png'
#创建ImageHelper()对象
ih1 = ImageHelper().read(gray_img)
ih2 = ImageHelper().read(color_img)
···
ihn = ImageHelper().read(xxx)
#创建ImageCollection并添加ih(或者通过ImageCollection.add_image()添加)
ic = ImageCollection(ih1, ih2,···，ihn)
#得到集合内所有图像按channel叠加的数据
data = ic.contact_with_channel()
```
