# ffun.DataProvider  
此模块主要负责数据提供  
- 导入package  
```python
from ffun.DataProvider import*
```  
## BatchHelper
- batch乱序  
batch-data在输入网络时最好是先经过乱序，这样训练处的模型鲁棒性会更强。

```python  
#数据batch和label序列，需要保证长度一致，可以是list或者tuple类型
batch = [1,2,3,4,5,6,7]
label = [1,2,3,4,5,6,7]
#实例化BatchHelper
bh = BatchHelper((batch,label))
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

## DataProvider  
DataProvider的职责是提供数据。在网络训练、测试中，所谓的数据即是用于训练或测试的数据以及标签。它有如下特点：

1. 是对BatchHelper的包装  
这点体现在它用组合的方式持有BatchHelper并对外限制API的输出能力。最重要的是输出next_batch()方法，num()方法等，而这些方法在BatchHelper中都有。

2. 是对BatchHelper的拓展  
它的next_batch()方法提供了分步加载接口。这个功能的实现，需要一个子类派生DataProvider类，并提供分步加载的load()方法。

基于以上两点，对DataProvider的使用有2种方式。一种是当做阉割版的BatchHelper使用，另一种是在load()中定制分步加载、数据预处理，或者再集成其他的方法作为一个超级数据提供者来用。尤其后者是一种高级用法。从解决问题的角度，DataProvider能解决的问题，用BatchHelper也可以解决，然而之所以提供一个独立的类而不是直接使用BatchHelper，是对网络训练数据输入提供一种在不失灵活性情况下的标准化规范。

### 用法1：阉割版BatchHelper

```python
# 1.原始数据a,b.实际使用可以运行时再确定数据
a = (1, 2, 3)
b = (4, 5, 6)
# 2.得到BatchHelper.入参为数据元组
bh = BatchHelper((a, b))
bh.shuffle(5)#乱序5次。也可以不乱序，注释这条代码
# 3.得到DataProvider.入参为bh，batch-size，加载模式。由于bh持有的数据，均为数据实体，所有使用once模式
dp = DataProvider(bh, 50, mode='once')
···
# 4.在其他地方使用dp
datas = dp.next_batch()
data, label = datas
```

### 用法2：高级使用

```python
class DataSource(DataProvider):
    'provide data、palceholder and feed_dict'
    def __init__(self, bh=None, batch_size=50, mode='once'):
        '''
        Input:
        - mode:'once'数据内容一次加载至内存，'part':分步加载至内存
        '''
        super(DataSource, self).__init__(bh, batch_size, mode)
        self.IMAGES_PL = None
        self.KEEP_PROP_PL = None
        self.LABELS_PL = None
        self.PL_OK = False
    def get_placeholder(self):
        '获得palceholder'
        if self.PL_OK:
            return self.IMAGES_PL, self.LABELS_PL, self.KEEP_PROP_PL
        H, W, C = CFG.Input_H, CFG.Input_W, CFG.Input_C
        self.IMAGES_PL = tf.placeholder(tf.float32, shape=(self.batch_size(), H, W, C))
        self.LABELS_PL = tf.placeholder(tf.int32, shape=(self.batch_size()))
        self.KEEP_PROP_PL = tf.placeholder('float')
        self.PL_OK = True
        return self.IMAGES_PL, self.LABELS_PL, self.KEEP_PROP_PL
    def get_feeddict(self, mode='train'):
        '获得feeddict'
        self.get_placeholder()
        prop = 0.5
        if mode == 'test':
            prop = 1.0
        images_feed, labels_feed = self.next_batch()#获得数据
        feed_dict = {
            self.IMAGES_PL: images_feed,
            self.LABELS_PL: labels_feed,
            self.KEEP_PROP_PL: prop
        }
        return feed_dict
    #覆盖父类的load函数
    def load(self):
        xxxxxxxx
        xxxxxxxx
        data = xxxx
        return data
```

如上所述，继承父类DataProvider，增加了placeholder、feeddict等方法，为网络提供数据。由于采用了'once'模式，load()方法其实不需要提供，因为只有在'part'模式下，DataProvider才会调用load()产生next_batch()的数据。以上仅是示例。
