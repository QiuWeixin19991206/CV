import tensorflow as tf
b = tf.fill([2,2],2.)
c = tf.ones([2,2])

b // c
b % c

tf.math.log(c)
tf.exp(c)
#没有log的8为底2为指数的函数，只有进行相除才能得到。

tf.math.log(8.) // tf.math.log(2.) #两个数还必须是浮点数

#矩阵的乘法
tf.pow(b,3)
b@c
tf.matmul(b,c)




print()