# WGAN的损失函数与GP（梯度惩罚）

import tensorflow as tf


def get_wgan_loss_fn():
    def d_loss_fn(reak_logit, fake_logit):
        return -tf.reduce_mean(reak_logit) + tf.reduce_mean(fake_logit)

    def g_loss_fn(fake_logit):
        return -tf.reduce_mean(fake_logit)
    return d_loss_fn, g_loss_fn


def gradient_penalty(f, real, fake):
    '''
    :param f:  判别器模型
    :param real:  真实图片
    :param fake:  生成图片
    :return:  gp项
    '''

    def intrepolate(x_real, x_fake):  # 为了计算GP，获取真实图片与生成图片之间的插值图片
        alpha_shape = [tf.shape(x_real)[0]] + [1] * (x_real.shape.ndims - 1)  # 插值系数
        alpha = tf.random.uniform(shape=alpha_shape, minval=0., maxval=1.)
        x_inter = alpha * x_real + (1 - alpha) * x_fake
        x_inter.set_shape(x_real.shape)
        return x_inter

    x = intrepolate(real, fake)
    with tf.GradientTape() as tape:
        tape.watch(x)  # 对常量张量求导需要增加watch
        pred = f(x)
    x_grad = tape.gradient(pred, x)  # 生成器D对插值图片X进行求导
    norm = tf.norm(tf.reshape(x_grad, [tf.shape(x_grad)[0], -1]), axis=1)  # 计算梯度的1范式
    gp = tf.reduce_mean((norm - 1.) ** 2)
    return gp









