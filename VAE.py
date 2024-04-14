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

class VAE(keras.Model):

    def __init__(self):
        super(VAE, self).__init__()

        # Encoders
        self.fc1 = layers.Dense(128)
        self.fc2 = layers.Dense(z_dim)
        self.fc3 = layers.Dense(z_dim)

        # Decoders
        self.fc4 = layers.Dense(128)
        self.fc5 = layers.Dense(784)#28 * 28

    def encoder(self, x):

        h = tf.nn.relu(self.fc1(x))
        # get mean
        mu = self.fc2(h)#平均值
        # get variance
        log_val = self.fc3(h)#方差

        return mu, log_val

    def decoder(self, z):

        out = tf.nn.relu(self.fc4(z))
        out = self.fc5(out)

        return out

    def reparameterization(self, mu, log_val):

        eps = tf.random.normal(log_val.shape)

        std = tf.exp(log_val*0.5)

        z = mu + std * eps
        return z

    def call(self, inputs, training=None, mask=None):

        mu, log_val = self.encoder(inputs)
        # reparameterization trick
        z = self.reparameterization(mu, log_val)

        x_hat = self.decoder(z)

        return x_hat, mu, log_val


if __name__ == '__main__':
    z_dim = 10
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

    model = VAE()
    model.build(input_shape=(4, 784))
    model.summary()
    optimizer = keras.optimizers.Adam(lr=lr)
    for epoch in range(100):

        for step, x in enumerate(train_db):

            #[b, 28, 28] => [b, 784]
            x = tf.reshape(x, [-1, 28*28])

            with tf.GradientTape() as tape:
                x_rec_logits, mu, log_var = model(x)

                rec_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=x, logits=x_rec_logits)#每一个像素当分类问题
                rec_loss = tf.reduce_sum(rec_loss) / x.shape[0]

                # compute kl divergence (mu, val) ~ N(0, 1)
                kl_div = -0.5 * (log_var + 1 - mu**2 - tf.exp(log_var))#log_var为log(sigma^2)
                kl_div = tf.reduce_sum(kl_div) / x.shape[0]

                loss = rec_loss + 1. * kl_div

            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            if step % 50 ==0:
                print(epoch, step, ' kl loss: ', float(kl_div), ' rec loss: ', float(rec_loss))


        # #evaluation
        z = tf.random.normal((batch_size, z_dim))
        logits = model.decoder(z)
        x_hat = tf.sigmoid(logits)
        x_hat = tf.reshape(x_hat, [-1, 28, 28]).numpy() * 255.
        x_hat = x_hat.astype(np.uint8)
        save_images(x_hat, 'vae_images/sampled_epoch_%d.png' % epoch)

        x = next(iter(test_db))
        x = tf.reshape(x, [-1, 784])
        x_hat_logits, _, _ = model(x)
        x_hat = tf.sigmoid(x_hat_logits)
        x_hat = tf.reshape(x_hat, [-1, 28*28]).numpy() * 255.
        x_hat = x_hat.astype(np.uint8)
        print(x_hat.shape)
        save_images(x_hat, 'vae_images/rec_epoch_%d.png' % epoch)

