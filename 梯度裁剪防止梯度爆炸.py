import tensorflow as tf


with tf.GradientTape() as tape:
    logits = model(x)
    loss = crteon(y, logits)

grads = tape.gradient(loss, model.trainable_variables)
grads = [tf.clip_by_global_norm(g, 15) for g in grads]#裁剪梯度在0-15之间一般采用10 15 20 防止梯度爆炸
optimizer.apply_gradients(zip(grads, model.trainable_variables))


