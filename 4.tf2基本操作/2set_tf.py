import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
np.random.seed(666)
x = np.random.uniform(-3,3,size=100)
x = x.reshape(-1,1)
y = 0.5 * x  + 3. + np.random.normal(size=(100,1))
# 如何建立一个tf，可以先建立np 然后将数据转化为一个tf 后在运行。

x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=666)
x_train = tf.convert_to_tensor(x_train).gpu()
y_train = tf.convert_to_tensor(y_train).gpu()

a1 =  tf.convert_to_tensor([1,2.]) #数据加工自动转化为一个tf
a2 = tf.zeros([2,3,5]) #创建一个0矩阵
a3 = tf.convert_to_tensor(np.random.random(size=(100,1)))
a4 = tf.convert_to_tensor(np.random.normal(0,1,size=(100,100)))
a5 = tf.convert_to_tensor(np.linspace(0,100,50))
a6 = tf.convert_to_tensor(np.empty([1,2]))
a7 = tf.convert_to_tensor(np.random.uniform(-3,3,size=(100,1)))#均匀分布
a7 = tf.convert_to_tensor(np.ones((2,3)))
a8 = tf.fill((2,2),9)
a9 = tf.random.normal((2,4),mean=0,stddev=1)
a10 = tf.random.truncated_normal((2,4),mean=0,stddev=1) #截断后的函数
a11 = tf.random.uniform((2,3),minval=10,maxval=100,dtype=float)
a12 = tf.random.shuffle(a11) #进行数据的打乱
# x = []
# y = []
# X = tf.gather(x,a12)  #得到打乱后的训练集和数据集 gather是一个切片操作
# Y = tf.gather(y,a12)  #得到打乱后的标签

x1 = tf.constant([[1,2,3],
                  [2,3,4]])
x2 = tf.zeros_like(x1) #生成和x1维度相同的矩阵
x2_1 = tf.zeros(x1.shape) #和x2的代码是等价的


print(x2)
print(x2_1)







