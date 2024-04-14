import tensorflow as tf
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics
from tensorflow import keras
import  os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.random.set_seed(2345)

class BasicBlock(layers.Layer):

    def __init__(self, filter_num, strides=1):
        super(BasicBlock, self).__init__()

        self.conv1 = layers.Conv2D(filter_num, (3, 3), strides=strides, padding='same')
        self.bn1 = layers.BatchNormalization()
        self.relu1 = layers.Activation('relu')

        self.conv2 = layers.Conv2D(filter_num, (3, 3), strides=1, padding='same')
        self.bn2 = layers.BatchNormalization()
        self.relu2 = layers.Activation('relu')
        if strides != 1:#若conv1进行了下采样 则为了残差相加形状一致，也得进行相同下采样
            self.downsample = Sequential()
            self.downsample.add(layers.Conv2D(filter_num, (1, 1), strides=strides))
        else:
            self.downsample = lambda x:x

    def call(self, inputs, training=None):

        # [b, h, w, c]
        conv1 = self.conv1(inputs)
        bn1 = self.bn1(conv1)
        relu1 = self.relu1(bn1)
        conv2 = self.conv2(relu1)
        bn2 = self.bn2(conv2)

        identity = self.downsample(inputs)

        add = layers.add([bn2, identity])#shortcut
        output = self.relu2(add)

        return output

class ResNet(keras.Model):
    def __init__(self,layer_dims, num_classes=100):#layer_dims [2, 2, 2, 2] num_classes=100全连接层有100类
        super(ResNet, self).__init__()
        #layer_1 预处理层
        self.layer_1 = Sequential([
            layers.Conv2D(64, kernel_size=3, strides=(1, 1), padding='same'),#(3, 3)
            layers.BatchNormalization(axis=-1, center=True, scale=True, trainable=True),
            layers.Activation('relu'),
            layers.MaxPool2D(pool_size=(2, 2), strides=(1, 1), padding='same')
        ])

        self.layer_2 = self.build_resblock(filter_num=64, blocks=layer_dims[0])
        self.layer_3 = self.build_resblock(filter_num=128, blocks=layer_dims[1], strides=2)
        self.layer_4 = self.build_resblock(filter_num=256, blocks=layer_dims[2], strides=2)
        self.layer_5 = self.build_resblock(filter_num=512, blocks=layer_dims[3], strides=2)
        # output: [b, 512, h, w] 不确定h, w 此层可以自适应来确定输出
        self.layer_6 = layers.GlobalAveragePooling2D()#全局平均池化
        self.layer_7 = layers.Flatten()
        self.layer_8 = layers.Dense(num_classes)

    def call(self, inputs, training=None, mask=None):
        x = self.layer_1(inputs)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)
        x = self.layer_5(x)
        x = self.layer_6(x)
        x = self.layer_7(x)
        x = self.layer_8(x)

        return x


    def build_resblock(self, filter_num, blocks, strides=1):#filter_num神经元数, blocks几个残差块
        res_blocks = Sequential()
        #可能进行下采样
        res_blocks.add(BasicBlock(filter_num, strides))#第一个残差块可以进行下采样

        for _ in range(1, blocks):
            res_blocks.add(BasicBlock(filter_num, 1))#其余不可以进行下采样
        return res_blocks

def resnet18():
    return ResNet([2, 2, 2, 2])#16 + 1fc + 1预处理层 = 18

def resnet34():
    return ResNet([3, 4, 6, 3])#(3 + 4 + 6 + 3) * 2  + 1fc + 1预处理层 = 34


def preprocess(x, y):
    # 对数据的预处理将数据控制在方差为1均值为0的范围中,归一化，将y编码
    x = tf.cast(x, dtype=tf.float32)  # / 255.
    y = tf.cast(y, dtype=tf.int32)

    mean = tf.reduce_mean(x)
    std = tf.math.reduce_std(x)
    x = (x - mean) / std
    y = tf.squeeze(y, axis=1)
    y = tf.one_hot(y, depth=100)
    return x, y


def data_read():
    (x_train, y_train), (x_test, y_test) = datasets.cifar100.load_data()
    print('data:', x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    x_train, y_train = preprocess(x_train, y_train)
    x_test, y_test = preprocess(x_test, y_test)
    print('preprocess:', x_train.shape, y_train.shape, x_test.shape, y_test.shape)

    train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_db = train_db.shuffle(1000).batch(64)
    test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_db = test_db.batch(64)

    sample = next(iter(train_db))
    print("sample", sample[0].shape, sample[1].shape, tf.reduce_min(sample[0]), tf.reduce_max(sample[1]))

    return train_db, test_db

if __name__=='__main__':
    train_db, test_db = data_read()  # 读取数据
    # 下面是快捷训练模型的keras接口
    # ------------------------------------------------------------------------------------
    model_conv = resnet18()  # 读取模型
    model_conv.build(input_shape=(None, 32, 32, 3))
    model_conv.summary()

    model_conv.compile(
        optimizer=optimizers.Adam(learning_rate=0.0001),
        loss=tf.losses.CategoricalCrossentropy(from_logits=True),
        # from_logits=True 表示该函数将对输入的 logits 进行处理，而不是对 softmax 函数的输出进行处理
        metrics=(["accuracy"])
    )
    model_conv.fit(train_db, epochs=1, validation_data=test_db, validation_freq=2)
    model_conv.evaluate(test_db)

    '''保存模型'''
    model_conv.save_weights("./ckpt/weights.ckpt")
    print("saved to ckpt/weights.ckpt")
    del model_conv
    '''加载模型'''
    model_conv = resnet18()
    model_conv.compile(
        optimizer=optimizers.Adam(learning_rate=0.0001),
        loss=tf.losses.CategoricalCrossentropy(from_logits=True),
        metrics=(["accuracy"])
    )
    model_conv.load_weights("./ckpt/weights.ckpt")
    print("loaded w form file")
    model_conv.evaluate(test_db)












