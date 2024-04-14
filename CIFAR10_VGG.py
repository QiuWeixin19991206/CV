import tensorflow as tf
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics
from tensorflow import keras
import  os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.random.set_seed(2345)

conv_layers = [# 5 units of conv + max pooling
    # unit 1
    layers.Conv2D(64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.Conv2D(64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.MaxPool2D(pool_size=[2, 2], strides=2, padding="same"),

    # unit 2
    layers.Conv2D(128, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.Conv2D(128, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.MaxPool2D(pool_size=[2, 2], strides=2, padding="same"),

    # unit 3
    layers.Conv2D(256, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.Conv2D(256, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.MaxPool2D(pool_size=[2, 2], strides=2, padding="same"),

    # unit 4
    layers.Conv2D(512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.Conv2D(512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.MaxPool2D(pool_size=[2, 2], strides=2, padding="same"),

    # unit 5
    layers.Conv2D(512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.Conv2D(512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.MaxPool2D(pool_size=[2, 2], strides=2, padding="same"),
]

def preprocess(x, y):
    x = tf.cast(x, dtype=tf.float32)# / 255.
    y = tf.cast(y, dtype=tf.int32)

    mean = tf.reduce_mean(x)
    std = tf.math.reduce_std(x)
    x = (x - mean) / std
    y = tf.squeeze(y, axis=1)

    return x, y

def data_read():
    (x_train, y_train), (x_test, y_test) = datasets.cifar100.load_data()
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    x_train, y_train = preprocess(x_train, y_train)
    x_test, y_test = preprocess(x_test, y_test)
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_db = train_db.shuffle(1000).batch(64)
    test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_db = test_db.batch(64)

    sample = next(iter(train_db))
    print("sample", sample[0].shape, sample[1].shape, tf.reduce_min(sample[0]), tf.reduce_max(sample[1]))

    return train_db, test_db

def main():
    train_db, test_db = data_read()  # 读取数据

    #1.分为两个Sequential容器建立模型
    conv_net = Sequential(conv_layers)
    # x = tf.random.normal([4, 32, 32, 3])
    # out = conv_net(x)
    # print(out.shape)
    fc_net = Sequential([
        layers.Dense(256, activation=tf.nn.relu),
        layers.Dense(128, activation=tf.nn.relu),
        layers.Dense(100, activation=None),
    ])

    conv_net.build(input_shape=[None, 32, 32, 3])
    fc_net.build(input_shape=[None, 512])

    optimizer = optimizers.Adam(learning_rate=1e-4)
    variables = conv_net.trainable_variables + fc_net.trainable_variables#两个网络没有用一个容器装，所以要把w b加在一起
    for epoch in range(1):

        for step, (x, y) in enumerate(train_db):

            with tf.GradientTape() as tape:
                # [b, 32, 32, 3] => [b, 1, 1, 512]
                out = conv_net(x)
                # flatten, => [b, 512]
                out = tf.reshape(out, [-1, 512])
                # [b, 512] => [b, 100]
                logits = fc_net(out)
                # [b] => [b, 100]
                y = tf.one_hot(y, depth=100)
                # compute loss
                loss = tf.losses.categorical_crossentropy(y, logits, from_logits=True)
                loss = tf.reduce_mean(loss)
            grads = tape.gradient(loss, variables)
            optimizer.apply_gradients(zip(grads, variables))

            if step %100 == 0:
                print(epoch, step, 'loss:', float(loss))

        total_correct = 0
        total_num = 0
        for x,y in test_db:
            out = conv_net(x)
            out = tf.reshape(out, [-1, 512])
            logits = fc_net(out)
            prob = tf.nn.softmax(logits, axis=1)
            pred = tf.argmax(prob, axis=1)
            pred = tf.cast(pred, dtype=tf.int32)

            correct = tf.cast(tf.equal(pred, y), dtype=tf.int32)
            correct = tf.reduce_sum(correct)

            total_num += x.shape[0]
            total_correct +=int(correct)
        acc = total_correct / total_num
        print(epoch, 'acc:', acc)
        
    #2.下面是快捷训练模型的keras接口
    #------------------------------------------------------------------------------------
    # model_conv = mod()  # 读取模型
    # model_conv.compile(
    #     optimizer=optimizers.Adam(learning_rate=0.0001),
    #     loss=tf.losses.CategoricalCrossentropy(from_logits=True),
    #     metrics=(["accuracy"])
    # )
    # model_conv.fit(train_db, epochs=30, validation_data=test_db, validation_freq=5)
    # model_conv.evaluate(test_db)
    # '''保存模型'''
    # model_conv.save_weights("./ckpt/weights.ckpt")
    # print("saved to ckpt/weights.ckpt")
    # del model_conv
    # '''加载模型'''
    # model = mod()
    # model.compile(
    #     optimizer=optimizers.Adam(learning_rate=0.0001),
    #     loss=tf.losses.CategoricalCrossentropy(from_logits=True),
    #     metrics=(["accuracy"])
    # )
    # model.load_weights("./ckpt/weights.ckpt")
    # print("loaded w form file")
    # model.evaluate(test_db)



def mod():
    #本质是一个vgg模型, 合并线性层和卷积层
    conv_layers = [

        layers.Conv2D(64,kernel_size=3,strides=1,padding="same",activation="relu"),
        layers.Conv2D(64, kernel_size=3, strides=1, padding="same", activation="relu"),
        layers.MaxPool2D(pool_size=2,strides=2,padding="same"),
        layers.BatchNormalization(axis=-1,center=True,scale=True,trainable=True),
        layers.Dropout(0.5),

        layers.Conv2D(128, kernel_size=3, strides=1, padding="same", activation="relu"),
        layers.Conv2D(128, kernel_size=3, strides=1, padding="same", activation="relu"),
        layers.MaxPool2D(pool_size=2, strides=2, padding="same"),
        layers.BatchNormalization(axis=-1,center=True,scale=True,trainable=True),

        layers.Conv2D(256, kernel_size=3, strides=1, padding="same", activation="relu"),
        layers.Conv2D(256, kernel_size=3, strides=1, padding="same", activation="relu"),
        layers.MaxPool2D(pool_size=2, strides=2, padding="same"),
        layers.BatchNormalization(axis=-1,center=True,scale=True,trainable=True),

        layers.Conv2D(512, kernel_size=3, strides=1, padding="same", activation="relu"),
        layers.Conv2D(512, kernel_size=3, strides=1, padding="same", activation="relu"),
        layers.MaxPool2D(pool_size=2, strides=2, padding="same"),
        layers.BatchNormalization(axis=-1,center=True,scale=True,trainable=True),

        layers.Conv2D(512, kernel_size=3, strides=1, padding="same", activation="relu"),
        layers.Conv2D(512, kernel_size=3, strides=1, padding="same", activation="relu"),
        layers.MaxPool2D(pool_size=2, strides=2, padding="same"),
        layers.BatchNormalization(axis=-1,center=True,scale=True,trainable=True),

        layers.Flatten(),#在全连接前要展开一下

        layers.Dense(256, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(128, activation="relu"),
        layers.Dense(100, activation="relu")
    ]

    conv_model = keras.Sequential(conv_layers)
    conv_model.build([None,32,32,3])
    return conv_model


if __name__=='__main__':
    main()











