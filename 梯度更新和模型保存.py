'''全连接层'''
# import tensorflow as tf
# from tensorflow import keras
#
# x = tf.random.normal([2, 3])
# model = keras.Sequential([
#     keras.layers.Dense(2, activation='relu'),
#     keras.layers.Dense(2, activation='relu'),
#     keras.layers.Dense(2)
# ])
# model.build(input_shape=[None, 3])
# model.summary()
# for p in model.trainable_variables:
#     print(p.name, p.shape)
'''梯度下降'''
import tensorflow as tf
w = tf.Variable(1.0)
b = tf.Variable(2.0)
x = tf.Variable(3.0)
with tf.GradientTape() as t1:
    with tf.GradientTape() as t2:
        y = x * w + b
    dy_dw, dy_db = t2.gradient(y, [w, b])#y其实就是loss
dy_dw2 = t1.gradient(dy_dw, w)
print(dy_dw)
print(dy_db)
print(dy_dw2)
assert dy_dw.numpy() == 3.0
assert dy_dw2 is None

'''反向传播算法推导'''




'''保存与加载模型'''
network.save_weights('weights.ckpt')
print('saved weights.')
del network

network = Sequential([layers.Dense(256, activation='relu'),
                     layers.Dense(128, activation='relu'),
                     layers.Dense(64, activation='relu'),
                     layers.Dense(32, activation='relu'),
                     layers.Dense(10)])
network.compile(optimizer=optimizers.Adam(lr=0.01),
		loss=tf.losses.CategoricalCrossentropy(from_logits=True),
		metrics=['accuracy']
	)
network.load_weights('weights.ckpt')
print('loaded weights!')
network.evaluate(ds_val)



network.save('model.h5')
print('saved total model.')
del network

print('loaded model from file.')
network = tf.keras.models.load_model('model.h5', compile=False)
network.compile(optimizer=optimizers.Adam(lr=0.01),
        loss=tf.losses.CategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
x_val = tf.cast(x_val, dtype=tf.float32) / 255.
x_val = tf.reshape(x_val, [-1, 28*28])
y_val = tf.cast(y_val, dtype=tf.int32)
y_val = tf.one_hot(y_val, depth=10)
ds_val = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(128)
network.evaluate(ds_val)