import  os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
tf.random.set_seed(1234)
np.random.seed(1234)
assert tf.__version__.startswith('2.')
#----------------------------------------------------------
#RNN源代码
class MyRNN(keras.Model):

    def __init__(self, units):
        super(MyRNN, self).__init__()

        # [b, 64]
        self.state0 = [tf.zeros([batchsz, units])]
        self.state1 = [tf.zeros([batchsz, units])]#第二层cell
        # 总共有多少的单词数量，单词维度，句子有80个单词
        # [b, 80] => [b, 80, 100]
        self.embedding = layers.Embedding(total_words, embedding_len, input_length=max_review_len)
        #[b, 80, 100], h_dim 64
        #RNN: cell1, cell2, cell3
        #SimpleRNN
        self.rnn_cell0 = layers.SimpleRNNCell(units, dropout=0.2)
        # SimpleRNN 第二层cell
        self.rnn_cell1 = layers.SimpleRNNCell(units, dropout=0.2)#第二层cell

        #fc [b, 80, 100] => [b, 64] => [b, 1]
        self.fc = layers.Dense(1)

    def call(self, inputs, training=None):
        """
                net(x) net(x, training=True) :train mode
                net(x, training=False): test
                :param inputs: [b, 80]
                :param training:
                :return:
        """
        # [b, 80]
        x = inputs
        # embedding: [b, 80] => [b, 80, 100]
        x = self.embedding(x)
        # rnn cell compute
        # [b, 80, 100] => [b, 64]
        state0 = self.state0
        state1 = self.state1#第二层cell

        for word in tf.unstack(x, axis=1): # word: [b, 100]
            # h1 = x*Wxh+h0*Whh
            # out0: [b, 64]
            out0, state0 = self.rnn_cell0(word, state0, training)
            out1, state1 = self.rnn_cell1(out0, state1, training)#第二层cell

        # out: [b, 64] => [b, 1]
        x = self.fc(out1)
        # p(y is pos|x)
        prob = tf.sigmoid(x)

        return prob

#---------------------------------------------------------------------
#keras接口的RNN

class MyRNN2(keras.Model):

    def __init__(self, units):
        super(MyRNN2, self).__init__()

        # transform text to embedding representation
        # [b, 80] => [b, 80, 100]
        self.embedding = layers.Embedding(total_words, embedding_len,
                                          input_length=max_review_len)

        # [b, 80, 100] , h_dim: 64
        self.rnn = keras.Sequential([
            layers.SimpleRNN(units, dropout=0.5, return_sequences=True, unroll=True),
            layers.SimpleRNN(units, dropout=0.5, unroll=True)
        ])

        # fc, [b, 80, 100] => [b, 64] => [b, 1]
        self.outlayer = layers.Dense(1)

    def call(self, inputs, training=None):
        """
        net(x) net(x, training=True) :train mode
        net(x, training=False): test
        :param inputs: [b, 80]
        :param training:
        :return:
        """
        # [b, 80]
        x = inputs
        # embedding: [b, 80] => [b, 80, 100]
        x = self.embedding(x)
        # rnn cell compute
        # x: [b, 80, 100] => [b, 64]
        x = self.rnn(x)

        # out: [b, 64] => [b, 1]
        x = self.outlayer(x)
        # p(y is pos|x)
        prob = tf.sigmoid(x)

        return prob

def main():
    units = 64
    epochs = 4

    model = MyRNN(units)
    model.compile(optimizer = keras.optimizers.Adam(0.001),
                  loss = tf.losses.BinaryCrossentropy(),
                  metrics=['accuracy'])
    model.fit(db_train, epochs=epochs, validation_data=db_test)

    model.evaluate(db_test)

#---------------------------------------------------------------------

if __name__=='__main__':
    batchsz = 128

    total_words = 10000#常见单词
    max_review_len = 80#句子长度过长 截取掉后面部分
    embedding_len = 100#一个单词用一个100维表示
    (x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=total_words)
    x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=max_review_len)# x_train:[b, 80]
    x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=max_review_len)# x_test: [b, 80]

    db_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    db_train = db_train.shuffle(1000).batch(batchsz, drop_remainder=True)#最后一个batch没有满128所以丢弃掉
    db_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    db_test = db_test.batch(batchsz, drop_remainder=True)
    print('x_train shape:', x_train.shape, tf.reduce_max(y_train), tf.reduce_min(y_train))
    print('x_test shape:', x_test.shape)

    units = 64
    epochs = 1
    model = MyRNN(units)
    model.compile(optimizer=keras.optimizers.Adam(0.001),
                  loss=tf.losses.BinaryCrossentropy(),
                  metrics=['accuracy']
                  )
    model.fit(db_train, epochs=epochs, validation_data=db_test)

    model.evaluate(db_test)

