import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
a = tf.random.normal([2,28,28,3])
print(a.shape[0])

a = a.numpy()
a.reshape(2,-1)
a.reshape(2,28,-1)
a.reshape(-1,3) #变化通道
#如果是tf的格式将使用,tf.reshape(s,[2,-1])
tf.reshape(a,[4,28,28,3]) #将其转化为一个[4,-1]后恢复成[4,28,28,3],但是要有实际意义
#矩阵的转置

tf.transport()#（h,w）--> (w,h)
a = tf.constant([1,2,3,4])
tf.transpose(a,perm=[0,2,1,3]).shape#output: [1,3,2,4] #该方案可用于pytorch和tf之间的数据的交互

print()
