import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
a = tf.random.normal([4,1,1,3])
#broadcasting 在前面添加一个维度，让两个tf数据维度相等 在一个维度复制自身，在维度为1可以复制
#broadcasting 是自动的判断的这个算法，
a1 = tf.broadcast_to(a,[4,32,32,3]) #将a进行扩张
b1 = tf.broadcast_to(a,[4,8,8,3])
b2 = tf.broadcast_to(a,[4,1,1,3])
#使用的维度的扩张tile,没有数据的优化，使用的内存空间是很大的
print()

