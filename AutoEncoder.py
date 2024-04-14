import tensorflow as tf
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics
from tensorflow import keras
import matplotlib.pyplot as plt
from  PIL import Image
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.random.set_seed(1234)
np.random.seed(1234)
assert tf.__version__.startswith('2.')

def save_images(imgs, name):#多张img拼为一张
    new_im = Image.new('L', (280, 280))

    index = 0
    for i in range(0, 280, 28):
        for j in range(0, 280, 28):
            im = imgs[index]
            im = Image.fromarray(im, mode='L')
            new_im.paste(im, (i, j))
            index += 1

    new_im.save(name)

class AE(keras.Model):

    def __init__(self):
        super(AE, self).__init__()

        # Encoders
        self.encoder = Sequential([
            layers.Dense(256, activation=tf.nn.relu),
            layers.Dense(128, activation=tf.nn.relu),
            layers.Dense(h_dim)
        ])
        # Decoders
        self.decoder = Sequential([
            layers.Dense(128, activation=tf.nn.relu),
            layers.Dense(256, activation=tf.nn.relu),
            layers.Dense(784)#还原 28，28
        ])

    def call(self, inputs, training=None, mask=None):
        # [b, 784] => [b, 10]
        h = self.encoder(inputs)
        # [b, 10] => [b, 784]
        x_hat = self.decoder(h)

        return x_hat

if __name__ == '__main__':
    h_dim = 20
    batch_size = 512
    lr = 1e-3

    (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
    x_train, x_test = x_train.astype(np.float32) / 255., x_test.astype(np.float32) / 255.#压缩到0-1区间
    #do not need label
    train_db = tf.data.Dataset.from_tensor_slices(x_train)
    train_db = train_db.shuffle(batch_size * 5).batch(batch_size)
    test_db = tf.data.Dataset.from_tensor_slices(x_test)
    test_db = test_db.batch(batch_size)

    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

    model = AE()
    # model.compile()中Keras 会自动根据模型的结构和输入形状来构建模型的权重。因此，并不是所有情况下都需要显式调用 model.build() 方法
    # 一般model.build()会搭配自定义 for .... with tf.GradientTape() as tape: 使用
    model.build(input_shape=(None, 784))
    model.summary()
    optimizer = keras.optimizers.Adam(lr=lr)
    for epoch in range(10):

        for step, x in enumerate(train_db):

            #[b, 28, 28] => [b, 784]
            x = tf.reshape(x, [-1, 28*28])

            with tf.GradientTape() as tape:
                x_rec_logits = model(x)

                rec_loss = tf.losses.binary_crossentropy(x, x_rec_logits, from_logits=True)#每一个像素当分类问题
                rec_loss = tf.reduce_mean(rec_loss)

            grads = tape.gradient(rec_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            if step % 50 ==0:
                print(epoch, step, float(rec_loss))


        #evaluation
        x = next(iter(test_db))
        logits = model(tf.reshape(x, [-1, 784]))
        x_hat = tf.sigmoid(logits)
        # [b, 784] => [b, 28, 28]
        x_hat = tf.reshape(x_hat, [-1, 28, 28])

        # [b, 28, 28] => [2b, 28, 28]
        x_concat = tf.concat([x, x_hat], axis=0)

        x_concat = x_concat.numpy() * 255.
        x_concat = x_concat.astype(np.uint8)#还原成numpy保存图片的格式了
        save_images(x_concat, 'ae_images/rec_epoch_%d.png' % epoch)







