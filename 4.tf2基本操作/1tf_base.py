import numpy as np
import tensorflow as tf

'''
#创建一个constant量 ，意思就是不会变的量
tf.constant(1.23)
tf.constant(1)
tf.constant(0.090)
tf.constant(1,dtype=float)
tf.constant([True,False])
tf.constant("String")
'''

'''
#使用gpu或者使用cpu的运算,以及它们之间的转换
with tf.device("cpu"):
    a = tf.constant(4,5)
with tf.device("gpu"):
    b = tf.range(4)
aa = tf.identity(a,"gpu") #将aa 使用的是gpu运算
bb = tf.identity(b,"gpu")
print(aa)
'''

'''
with tf.device("cpu"):
    a = tf.constant([[1,3],
                     [2,4]])
with tf.device("gpu"):
    b = tf.range(4)
aa = a.numpy() #将tf转化为np
#ndim 和shape 在tf和np都可以用
print(aa.ndim )
print(b.ndim )
#可以得到维度
print(aa.shape )
print(b.shape )
print(tf.rank(b)) #tf中的数的维度，不常用知道就行
'''

'''
#判断是不是tf的类型
with tf.device("cpu"):
    a = tf.constant(1)
with tf.device("gpu"):
    b = tf.constant(3)
c = np.arange(2)

print(a + b)
print(tf.is_tensor(c)) #判断是不是tf的类型
print(tf.is_tensor(a))
print(a.dtype,b.dtype,c.dtype)

#tf 中数据的转化的函数
d = np.arange(5)
print(d.dtype)
e = tf.convert_to_tensor(d,dtype=tf.int64) #将数据转化为一个tf
print(e.dtype)
E = tf.cast(a,dtype=tf.float32) #关于数据类型的转换
print(E.dtype)
'''

'''
# tf 中bool的转化(通过cast函数事项类型的相互的转换)
a = tf.constant([0,0,1,1])
print(a.dtype)
a = tf.cast(a,dtype=tf.bool)
print(a)
print(a.dtype)
a = tf.cast(a,tf.float32)
print(a)
print(a.dtype)
'''

a = tf.range(5)
b = tf.Variable(a,name="input") #指定神经网络中国求梯度的参数
c = tf.constant(1)
print(b.dtype)
print(b.name)
print(b.trainable) #是否包含梯度信息

#tf数据支持强制转换
print(int(c))
print(float(c))
