import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets
from sklearn.metrics import mean_squared_error
import numpy as np
from scipy.io import loadmat
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def load_data():
    (x, y), (x_test, y_test) = keras.datasets.mnist.load_data()
    x = tf.convert_to_tensor(x,dtype=tf.float32)/255.
    y = tf.convert_to_tensor(y,dtype=tf.int32)
    x_test = tf.convert_to_tensor(x_test,dtype=tf.float32)
    y_test = tf.convert_to_tensor(y_test,dtype=tf.int32)

    train_db = tf.data.Dataset.from_tensor_slices((x,y)).batch(128)
    test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    train_db.shuffle(1000)
    train_db.batch(128)
    train_iter = iter(train_db)
    smple = next(train_iter)

    return train_db
# 变换过程[b,784]-->>[b,256]---->>[b,128]---->[b,10]
# w的定义[in,out] b的定义为[out]
with tf.device('/GPU:0'):
    w1 = tf.Variable(tf.random.truncated_normal([784, 256], stddev=0.1))
    b1 = tf.Variable(tf.zeros([256]))
    w2 = tf.Variable(tf.random.truncated_normal([256, 128], stddev=0.1))
    b2 = tf.Variable(tf.zeros([128]))
    w3 = tf.Variable(tf.random.truncated_normal([128, 10], stddev=0.1))
    b3 = tf.Variable(tf.zeros([10]))
    lr = 0.001
    for epoch in range(100):  # iterate db for 10
        for step, (x, y) in enumerate(load_data()):
            # x[128,28,28]
            # y:[128]
            x_t = x
            y_t = y
            x = tf.reshape(x, [-1, x.shape[1] * x.shape[2]])
            # x[128,28*28] 维度变换
            #包含一个隐层的神经网络
            with tf.GradientTape() as tape:  # 跟踪张量
                h1 = x @ w1 + b1
                #在这里有一个自动的broadcasting
                h1 = tf.nn.relu(h1)  # 激活函数
                # 第一次传播
                # [b,784] * [784,256] + [256] => [b,256] + [256]

                h2 = h1 @ w2 + b2
                h2 = tf.nn.relu(h2)
                # 第二次传播
                # [b,256]*[256,128] + [128] =》  [b,128] + [128]

                out = h2 @ w3 + b3
                # 第三次传播
                # [b,128]*[128,10] + [10] =》  [b,10] + [10]

                y_onehot = tf.one_hot(y, depth=10)  # 给这10个标签给编码
                # out:[b,10] y:[b]=>[b,10]

                loss = tf.square(y_onehot - out)
                loss = tf.reduce_mean(loss)
                # 使用均方误差
            grads = tape.gradient(loss, [w1, b1, w2, b2, w3, b3])
            w1.assign_sub(lr * grads[0])  # 这个操作会把一个张量的当前值减去另一个张量的值，并且将结果存回第一个张量。该操作返回一个操作对象，可用于在TensorFlow图形中执行操作。
            b1.assign_sub(lr * grads[1])
            w2.assign_sub(lr * grads[2])
            b2.assign_sub(lr * grads[3])
            w3.assign_sub(lr * grads[4])
            b3.assign_sub(lr * grads[5])

            if step % 100 == 0:
                print(epoch, step, 'loss:', float(loss))






