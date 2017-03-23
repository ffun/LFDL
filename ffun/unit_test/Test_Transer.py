
import Transer
import numpy as np

a = np.zeros((2,2,2))
b = np.full((2,2,2),3)

print 'a:\n',a
print 'a:\n',b

c = Transer.im_con_Channel((a,b))

print a.shape,b.shape,c.shape