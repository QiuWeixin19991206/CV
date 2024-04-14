import  os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import cv2 as cv
import matplotlib.pyplot as plt
tf.random.set_seed(1234)
np.random.seed(1234)
assert tf.__version__.startswith('2.')
#---------------------------------------------------------------------
#keras接口的RNN

class net(keras.Model):

    def __init__(self):
        super(net, self).__init__()


        # [b, 80, 100] , h_dim: 64
        self.layer1 = keras.Sequential([
            layers.MaxPool2D(pool_size=[2, 2], strides=2, padding="valid")
        ])


    def call(self, inputs, training=None, mask=None):

        x = inputs
        x = self.layer1(x)

        return x

def main():

    model = net()
    X = cv.imread('E:\c++\pantyhose.jpg')
    x = tf.convert_to_tensor(X)
    x = tf.reshape(x, (1, 1920, 1080, 3))
    y = model(x)
    print(x.shape, y.shape)
    img = y.numpy()
    img = np.squeeze(img)
    plt.imshow(img[:, :, ::-1])
    plt.show()
#---------------------------------------------------------------------

if __name__=='__main__':
    main()


