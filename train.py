import functools

import tensorflow as tf
tf.config.experimental.set_memory_growth(tf.config.list_physical_devices("GPU")[0], True)  # 设置GPU显存用量按需使用, 需要紧跟在import tf后面设置GPU显存，不然后面导入的包可能会实例化模型，造成显存分配失败

import os
import tqdm
import numpy as np

import config, dataset, loss, module
from utils import path_util, lr_decay_util, image_util


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
val_dataset, val_img_shape, len_val_dataset = dataset.make_celeba_dataset(config.IMG_DIR,
                                                                             config.VAL_LABEL_PATH,
                                                                             config.DEFAULT_ATT_NAMES,
                                                                             config.N_SAMPLE,
                                                                             config.LOAD_SIZE,
                                                                             config.CROP_SIZE,
                                                                             training=False)
n_atts = len(config.DEFAULT_ATT_NAMES)  # 训练或修改的特征数量


# ==============================================================================
# =                               2.   model                                   =
# ==============================================================================
G_enc = module.get_G_enc(input_shape=train_img_shape, n_downsamplings=config.N_DOWNSAMPLINGS, weight_decay=config.WEIGHT_DECAY)

zs_shape = [z.shape for z in G_enc.output]  # z表示G_enc每层的输出，共五层
G_dec = module.get_G_dec(zs_shape=zs_shape, atts_shape=(n_atts, ), n_upsamplings=config.N_UPSAMPLINGS, weight_decay=config.WEIGHT_DECAY)

D, C = module.get_D_and_C(n_atts=n_atts, input_shape=train_img_shape, n_downsamplings=config.N_DOWNSAMPLINGS, weight_decay=config.WEIGHT_DECAY)

d_loss_fn, g_loss_fn = loss.get_wgan_loss_fn()
lr_decay = lr_decay_util.LinearDecayLR(config.LEARNING_RATE, config.N_EPOCHS, config.EPOCH_START_DECAY)  # 学习率衰减函数，返回衰减后的学习率
# G_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_decay, beta_1=config.BATE_1)  # lr_decay.__call__中的step指的是epoch么
# D_and_C_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_decay, beta_1=config.BATE_1)
G_optimizer = tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE, beta_1=config.BATE_1)  # lr_decay.__call__中的step指的是epoch么
D_and_C_optimizer = tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE, beta_1=config.BATE_1)


# for x_a, atts_a in train_dataset.take(1):  验证传入数据到模型中，能有正常的输出
#     x_z = G_enc(x_a, training=False)
#     # print(x_z)
#     # print('=============================')
#     x_b = G_dec(x_z + [atts_a * 2 - 1], training=False)
#     # print('x_b  =============================')
#     # print(x_b)
#     print('D(x_a)  =============================')
#     print(D(x_a))
#     print('C(x_a)  =============================')
#     print(C(x_a))
#     print('D(x_b)  =============================')
#     print(D(x_b))
#     print('C(x_b)  =============================')
#     print(C(x_b))

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
        z_a = G_enc(x_a, training=True)
        x_a_hat = G_dec(z_a + [atts_a_], training=True)  # 重构的图片
        x_b = G_dec(z_a + [atts_b_], training=True)  # 具有特征b的生成图片

        # 重构损失
        G_rec_loss = tf.reduce_mean(tf.losses.mean_absolute_error(x_a, x_a_hat))

        # 对抗损失
        xb_logit_D = D(x_b, training=True)  # 生成图片在判别器的输出
        G_adv_loss = g_loss_fn(xb_logit_D)

        # 特征限制损失
        xb_logit_C = C(x_b, training=True)
        G_att_loss = tf.reduce_mean(tf.losses.binary_crossentropy(atts_b, xb_logit_C))

        G_loss = config.G_RECONSTRUCTION_LOSS_WEIGHT * G_rec_loss + config.G_ATTRIBUTE_LOSS_WEIGHT * G_att_loss + G_adv_loss
    G_gradients = tape.gradient(G_loss, [*G_enc.trainable_variables, *G_dec.trainable_variables])
    G_optimizer.apply_gradients(zip(G_gradients, [*G_enc.trainable_variables, *G_dec.trainable_variables]))

    return G_loss


# for x_a, atts_a in train_dataset.take(1):
#     print(train_step_G(x_a, atts_a))


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

        real_z = G_enc(x_a, training=True)
        x_b = G_dec(real_z + [atts_b], training=True)  # 具有特征b的生成图片

        # 判别器的损失函数
        xa_logit_D = D(x_a, training=True)
        xb_logit_D = D(x_b, training=True)
        wgan_d_loss = d_loss_fn(xa_logit_D, xb_logit_D)
        gp = loss.gradient_penalty(functools.partial(D, training=True), x_a, x_b)
        D_loss = wgan_d_loss + config.GP_WEIGHT * gp

        # 分类器的损失函数
        xa_logit_C = C(x_a, training=True)
        C_loss = tf.reduce_mean(tf.losses.binary_crossentropy(atts_a, xa_logit_C))  # 二分类损失函数

        # reg_loss = tf.reduce_sum(D.func.reg_losses)  # 源码中还加上了这个损失

        D_and_C_loss = D_loss + config.C_ATTRIBUTE_LOSS_WEIGHT * C_loss
    D_gradients = tape.gradient(D_and_C_loss, [*D.trainable_variables, *C.trainable_variables])
    D_and_C_optimizer.apply_gradients(zip(D_gradients, [*D.trainable_variables, *C.trainable_variables]))

    return D_and_C_loss


# for x_a, atts_a in train_dataset.take(1):
#     print(train_step_D(x_a, atts_a))


# ==============================================================================
# =                               3.5   sample                                 =
# ==============================================================================
val_x_a, val_atts_a = list(val_dataset.take(1))[0]  # 每次采样的图片固定取第一批
def sampel(sample_x_a, sample_atts_a, epoch, iter):
    atts_b_list = [sample_x_a]
    for i in range(n_atts):
        tmp = np.array(sample_atts_a, copy=True)
        tmp[:, i] = 1 - tmp[:, i]  # 修改特征标签，即原来是0的特征修改为1，原来是1的特征修改为0

        tmp = dataset.check_attribute_conflict(tmp, config.DEFAULT_ATT_NAMES[i], config.DEFAULT_ATT_NAMES)  # 检测需要修改的一个特征有没有和原来的特征冲突
        atts_b_list.append(tmp)

    x_b_list = [sample_x_a]  # 第一个元素是真实的图片集
    for i, atts_b in enumerate(atts_b_list[1:]):
        atts_b_ = tf.cast((atts_b * 2 - 1), tf.float32)  # 特征标签0/1 ==> -1/1
        # if i > 0:  # i==0时保留原来的真实图片
            # atts_b_[..., i - 1].assign(atts_b_[..., i - 1] * 2.0)
        x_b = G_dec(G_enc(sample_x_a, training=False) + [atts_b_], training=False)
        x_b_list.append(x_b)

    sample = np.transpose(x_b_list, (1, 2, 0, 3, 4))  # (len, batch, H, W, N_C) ==> (batch, H, len, W, N_C)
    sample = np.reshape(sample, (-1, sample.shape[2] * sample.shape[3], sample.shape[4]))
    image_util.imwrite(sample, '%s/Epoch-%d_Iter-%d.jpg' % (sample_dir, epoch, iter))


# def sample_test(datasets):
#     for x_a, atts_a in datasets.take(1):  # 每次采样的图片固定取第一批
#         atts_b_list = [atts_a]  # 第一个元素是对应图片的特征标签，值为0/1
#         for i in range(n_atts):
#             tmp = np.array(atts_a, copy=True)
#             tmp[:, i] = 1 - tmp[:, i]  # 修改特征标签，即原来是0的特征修改为1，原来是1的特征修改为0
#
#             tmp = dataset.check_attribute_conflict(tmp, config.DEFAULT_ATT_NAMES[i],
#                                                    config.DEFAULT_ATT_NAMES)  # 检测需要修改的一个特征有没有和原来的特征冲突
#             atts_b_list.append(tmp)  # tmp=修改单个特征之后的特征b, atts_b_list=[原来的特征标签， 修改第一个特征后的特征标签， ..., 修改第n_atts个特征后的特征标签]
#         return x_a, atts_b_list


# ==============================================================================
# =                               4.   train                                   =
# ==============================================================================
def train():
    for epoch in tqdm.trange(config.N_EPOCHS, desc='Epoch Loop'):
        for x_a, atts_a in tqdm.tqdm(train_dataset, desc='Batch Loop', total=len_train_dataset):
            D_and_C_loss = train_step_D(x_a, atts_a)

            if D_and_C_optimizer.iterations.numpy() % config.N_D == 0:  # 每训练N_D次判别器，训练一次生成器
                G_loss = train_step_G(x_a, atts_a)

            if G_optimizer.iterations.numpy() % 100 == 0:
                sampel(val_x_a, val_atts_a, epoch, G_optimizer.iterations.numpy())

        print('epoch:%d, g_loss:%f, d_loss:%f' % (epoch + 1, G_loss, D_and_C_loss))


# ==============================================================================
# =                               5.   save model                            =
# ==============================================================================
def save_model(models=[], names=[]):
    for i, model in enumerate(models):
        model.save('./model/' + names[i], save_format="tf")


# ==============================================================================
# =                               6.   run                                     =
# ==============================================================================
train()
save_model([G_enc, G_dec], ['G_enc', 'G_dec'])









