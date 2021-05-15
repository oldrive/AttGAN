import functools

import tensorflow as tf
tf.config.experimental.set_memory_growth(tf.config.list_physical_devices("GPU")[0], True)  # 设置GPU显存用量按需使用, 需要紧跟在import tf后面设置GPU显存，不然后面导入的包可能会实例化模型，造成显存分配失败

import os
import tqdm

import config, dataset, loss, module
from utils import path_util, lr_decay_util


# ==============================================================================
# =                               0.   output_dir                              =
# ==============================================================================
output_dir = os.path.join('output', 'celeba_attGAN')
path_util.mkdir(output_dir)
sample_dir = os.path.join(output_dir, 'samples_training')  # 训练过程中生成的图片目录
path_util.mkdir(sample_dir)


# ==============================================================================
# =                               1.   data                                    =
# ==============================================================================
train_dataset, train_img_shape, len_train_dataset = dataset.make_celeba_dataset(config.IMG_DIR,
                                                                             config.TRAIN_LABEL_PATH,
                                                                             config.DEFAULT_ATT_NAMES,
                                                                             config.BATCH_SIZE,
                                                                             config.LOAD_SIZE,
                                                                             config.CROP_SIZE)
n_atts = len(config.DEFAULT_ATT_NAMES)  # 训练或修改的特征数量


# ==============================================================================
# =                               2.   model                                   =
# ==============================================================================
G_enc = module.get_G_enc(input_shape=train_img_shape, n_downsamplings=config.N_DOWNSAMPLINGS, weight_decay=config.WEIGHT_DECAY)

zs_shape = [z.shape for z in G_enc.output]  # z表示G_enc每层的输出，共五层
G_dec = module.get_G_dec(zs_shape=zs_shape, atts_shape=(n_atts, ), n_upsamplings=config.N_UPSAMPLINGS, weight_decay=config.WEIGHT_DECAY)

D, C = module.get_D_and_C(n_atts=n_atts, input_shape=train_img_shape, n_downsamplings=config.N_DOWNSAMPLINGS, weight_decay=config.WEIGHT_DECAY)

d_loss_fn, g_loss_fn = loss.get_wgan_loss_fn()
G_optimizer = tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE, beta_1=config.BATE_1)
D_and_C_optimizer = tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE, beta_1=config.BATE_1)
# print(G_enc.summary())
# print(G_dec.summary())
# print(D.summary())
# print(C.summary())
# print('==========D==============')
# print(D.trainable_weights[0])
# print('==========C==============')
# print(C.trainable_weights[0])
# print('==========changed D==============')
# D.trainable_weights[0].assign(tf.clip_by_value(D.trainable_weights[0], -0.01, 0.01))  # D和C前面的层确实做到了参数共享
# print('==========D==============')
# print(D.trainable_weights[0])
# print('==========C==============')
# print(C.trainable_weights[0])
# for imgs_test, atts_test in train_dataset.take(1):  # 确保G_enc有五个输出
#     print(atts_test)
#     zs_test = G_enc(imgs_test)
#
#     G_dec(zs_test + [atts_test])  # 多输入的模型，在输入数据时需要传入列表进去
#     print(len(G_enc(imgs_test)))


# ==============================================================================
# =                               3.   train_step                              =
# ==============================================================================
@tf.function
def train_step_G(x_a, atts_a):
    with tf.GradientTape() as tape:
        atts_b = tf.random.shuffle(atts_a)  # 对应论文中的特征b
        atts_a_ = atts_a * 2 - 1
        atts_b_ = atts_b * 2 - 1  # 将值从 0/1 ==> -1/1， 因为送入到G_dec的特征的值要求为-1/1
        z_a = G_enc(x_a)
        x_a_hat = G_dec(z_a + [atts_a_])  # 重构的图片
        x_b = G_dec(z_a + [atts_b_])  # 具有特征b的生成图片

        # 重构损失
        G_rec_loss = tf.losses.mean_absolute_error(x_a, x_a_hat)

        # 对抗损失
        xb_logit_D = D(x_b)  # 生成图片在判别器的输出
        G_adv_loss = g_loss_fn(xb_logit_D)

        # 特征限制损失
        xb_logit_C = C(x_b)
        G_att_loss = tf.losses.binary_crossentropy(atts_b, xb_logit_C)

        G_loss = config.G_RECONSTRUCTION_LOSS_WEIGHT * G_rec_loss + config.G_ATTRIBUTE_LOSS_WEIGHT * G_att_loss + G_adv_loss
    G_gradients = tape.gradient(G_loss, G_enc.trainable_weights + G_dec.trainable_weights)
    G_optimizer.apply_gradients(zip(G_gradients, G_enc.trainable_weights + G_dec.trainable_weights))

    return G_loss


@tf.function
def train_step_D(x_a, atts_a):
    '''

    :param x_a:  具有特征a的真实图片
    :param atts_a:  对应着论文中的特征a，真实图片中具有的特征，值为0/1
    :return:
    '''
    with tf.GradientTape() as tape:
        atts_b = tf.random.shuffle(atts_a)  # 对应论文中的特征b
        atts_b = atts_b * 2 - 1  # 将值从 0/1 ==> -1/1

        real_z = G_enc(x_a)
        x_b = G_dec(real_z + [atts_b])  # 具有特征b的生成图片

        # 判别器的损失函数
        xa_logit_D = D(x_a)
        xb_logit_D = D(x_b)
        wgan_d_loss = d_loss_fn(xa_logit_D, xb_logit_D)
        gp = loss.gradient_penalty(functools.partial(D, training=True), x_a, x_b)
        D_loss = wgan_d_loss + config.GP_WEIGHT * gp

        # 分类器的损失函数
        xa_logit_C = C(x_a)
        C_loss = tf.losses.binary_crossentropy(atts_a, xa_logit_C)  # 二分类损失函数

        # reg_loss = tf.reduce_sum(D.func.reg_losses)  # 源码中还加上了这个损失

        D_and_C_loss = D_loss + config.C_ATTRIBUTE_LOSS_WEIGHT * C_loss
    D_gradients = tape.gradient(D_and_C_loss, D.trainable_weights + C.trainable_weights)
    D_and_C_optimizer.apply_gradients(zip(D_gradients, D.trainable_weights + C.trainable_weights))

    return D_and_C_loss


# ==============================================================================
# =                               4.   train                                   =
# ==============================================================================
def train():
    lr_fn = lr_decay_util.LinearDecayLR(config.LEARNING_RATE, config.N_EPOCHS, config.EPOCH_START_DECAY)  # 学习率衰减函数，返回衰减后的学习率
    for epoch in tqdm.trange(config.EPOCHS, desc='Epoch Loop'):
        lr_decayed = lr_fn(epoch)
        G_optimizer.weights









