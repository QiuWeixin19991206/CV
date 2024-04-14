import tensorflow as tf


with tf.GradientTape() as tape:
    logits = model(x)
    loss = crteon(y, logits)

grads = tape.gradient(loss, model.trainable_variables)
grads = [tf.clip_by_global_norm(g, 15) for g in grads]#裁剪梯度在0-15之间一般采用10 15 20 防止梯度爆炸
optimizer.apply_gradients(zip(grads, model.trainable_variables))


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


import datetime
def get_time():#获取当前时间
    current_time = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    return current_time
