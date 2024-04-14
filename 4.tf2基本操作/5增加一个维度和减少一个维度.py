import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
a = tf.random.normal([4,35,8])
tf.expand_dims(a,axis=0).shape #返回的维度[1,4,35,8]
tf.expand_dims(a,axis=3).shape #返回的维度[4,35,8,1] qi
b = tf.convert_to_tensor([3,1,4,6])
tf.squeeze(b,axis=1) #将b中的第二个维度的数据去掉 同样的axis可以为负数
