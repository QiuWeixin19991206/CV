import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

(x, y), (x_test, y_test) = datasets.mnist.load_data()
x = tf.convert_to_tensor(x, dtype=tf.float32)
y = tf.convert_to_tensor(y, dtype=tf.int32)
x_test = tf.convert_to_tensor(x_test, dtype=tf.float32)
y_test = tf.convert_to_tensor(y_test, dtype=tf.int32)
train_db = tf.data.Dataset.from_tensor_slices((x, y)).batch(128)
train_iter = iter(train_db)#迭代器
test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(128)#分开数据防止数据泄露
test_iter = iter(test_db)#迭代器
sample = next(train_iter)
#初始化模型 model = MyModel()
#初始化权重 model.build()
w1 = tf.Variable(tf.random.truncated_normal([784, 256], stddev=0.1))
b1 = tf.Variable(tf.zeros([256]))
w2 = tf.Variable(tf.random.truncated_normal([256, 128], stddev=0.1))
b2 = tf.Variable(tf.zeros([128]))
w3 = tf.Variable(tf.random.truncated_normal([128, 10], stddev=0.1))
b3 = tf.Variable(tf.zeros([10]))

lr = 1e-3
#前向运算
for epoch in range(10):#iterate db for 10
    for step, (x, y) in enumerate(train_db):#for every batch
        # x:[128,28,28] y:[128]
        x = tf.reshape(x, [-1, 28*28])
        with tf.GradientTape() as tape:#梯度信息记录下来
            #前向传播
            h1 = x@w1 + b1#[b, 784]@[784, 256] + [256] => [b, 256]
            h1 = tf.nn.relu(h1)
            h2 = h1@w2 + b2
            h2 = tf.nn.relu(h2)
            out = h2@w3 + b3
            #计算误差
            y_onehot = tf.one_hot(y, depth=10)#网络输出为10
            #计算loss
            loss = tf.square(y_onehot - out)
            loss = tf.reduce_mean(loss)#tensor: scalar
        #计算梯度 grads = tape.gradient(rec_loss, model.trainable_variables)
        grads = tape.gradient(loss, [w1, b1, w2, b2, w3, b3])
        grads, _ = tf.clip_by_global_norm(grads, 15)#约束权重限幅
        # w1 = w1 - lr * grads[0]
        # b1 = b1 - lr * grads[1]
        # w2 = w2 - lr * grads[2]
        # b2 = b2 - lr * grads[3]
        # w3 = w3 - lr * grads[4]
        # b3 = b3 - lr * grads[5]
        #原地梯度更新 保持类型 optimizer.apply_gradients(zip(grads, model.trainable_variables))
        w1.assign_sub(lr * grads[0])
        b1.assign_sub(lr * grads[1])
        w2.assign_sub(lr * grads[2])
        b2.assign_sub(lr * grads[3])
        w3.assign_sub(lr * grads[4])
        b3.assign_sub(lr * grads[5])

        if step % 100 == 0:
            print(epoch, step, 'loss', float(loss))
    #test/evluation
    total_correct, total_num = 0, 0
    for step, (x, y) in enumerate(test_db):
        x = tf.reshape(x, [-1, 28 * 28])
        h1 =tf.nn.relu(x@w1 + b1)
        h2 =tf.nn.relu(h1@w2 + b2)
        out = h2@w3 + b3
        prob = tf.nn.softmax(out, axis=1)#因为one-hot 是在0-1，使用softmax使之在0-1
        pred = tf.cast(tf.argmax(prob, axis=1), dtype=tf.int32)#axis=1指定列
        correct = tf.cast(tf.equal(pred, y), dtype=tf.int32)
        correct = tf.reduce_sum(correct)#求和为对的个数
        total_correct += int(correct)
        total_num += x.shape[0]
    acc = total_correct / total_num
    print('test acc:',acc)


# import tensorflow as tf
# y = tf.linspace(-2, 2 , 5)
# x = tf.linspace(-1, 1 , 3)
# print(x, y)
# x , y = tf.meshgrid(x, y)
# print(x, y)
# points = tf.stack([x, y], axis=2)
# print(points)

# import keras.datasets.mnist
# import tensorflow as tf
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# (x, y), (x_test, y_test) = keras.datasets.mnist.load_data()
# x = x.reshape(-1,x.shape[1] * x.shape[2])
# x_test = x_test.reshape(-1,x_test.shape[1] * x_test.shape[2])
# transfer = StandardScaler()
# x = transfer.fit_transform(x)
# x_test = transfer.transform(x_test)
# #将标签进行编码，减小内存空间的使用
# y_onehot_train = tf.one_hot(y,depth=10)
# print(y_onehot_train)
#
# import keras.datasets.mnist
# import tensorflow as tf
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# def prep(x, y):
#     return
# def mnist_dataset():
#     (x, y), (x_test, y_test) = keras.datasets.mnist.load_data()
#     y = tf.one_hot(y, depth=10)
#     y_val = tf.one_hot(y_test, depth=10)
#     ds = tf.data.Dataset.from_tensor_slices((x, y))
#     ds = ds.map(prep)# 数据的预处理
#     ds = ds.shuffle(60000).batch(100)
#     ds_val = tf.data.Dataset.from_tensor_slices((x_test, y_test))
#     ds_val = ds_val.map(prep)  # 数据的预处理
#     ds_val = ds_val.shuffle(10000).batch(100)
#     return ds, ds_val
#
# mnist_dataset()


