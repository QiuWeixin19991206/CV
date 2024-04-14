import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
#逐步的索引
a =tf.ones([2,5,5,3])
a[0][0]
a[0][0][0]
a[0][0][0][2]

#切片（可以用np索引的规则）
a[:,0]
a[:,:2,:2,2 ] #
a[:,:,:,1] #相当于取出所有的图片的绿色通道
#索引
a = tf.constant([4,25,8])

print(tf.gather(a,axis=0,indices=[2,3]).shape) #采样器不能使用的冒号？[2,25,8]
print(tf.gather_nd(a,[0])) #取第一个维度的所有数据 output：[35,8]
print(tf.gather_nd(a,[0,1])) #取的寄一个维度第一个数据中的第二个数据的所有的数据out:[8]
print(tf.gather_nd(a,[0,1,1])) #putput: [0]
print(tf.gather_nd(a,[[0,1,1]])) #putput: [1] 指定的单个数据,也就是单个样本[返回的是一个限量]
print(tf.gather_nd(a,[[0,0],[1,1]]))#output : [2,8]
print(tf.fateh_nd(a,[[[1,1,2],[1,1,1],[0,0,1]]]))# output : [1,3]
b = tf.constant([4,28,28,3])
print(tf.boolean_mask(b,mask = [True,True,False,False]).shape)#output:[2,28,28,3]
print(tf.boolean_mask(b,mask = [True,True,False],axis=3).shape)#output:[2,28,28,2] 取的是第三通道的r和g
c = tf.constant([2,4,3])
print(tf.boolean_mask(c,[[True,False,False],
                         [True,False,False]])) #返回的是[2,4]的数组
tf.gather_nd()
tf.boolean_mask()
