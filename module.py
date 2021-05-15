import tensorflow as tf
from tensorflow.keras import layers, Model
# import tensorflow_addons as tfa
# 如果使用tfa.layers.InstanceNormalization，在判别器和分类器的输出都会固定为0和[0.5, 0.5, ...]，原因不明

import config
import dataset
from utils import tile_util


# ==============================================================================
# =                        256x256x3 ==> G_enc ==> 8x8x1024                    =
# ==============================================================================
def get_G_enc(input_shape=(256, 256, 3), dim=64, n_downsamplings=5, weight_decay=0.0, name='G_enc'):
    zs = []  # 每一层的输出，用于与G_dec之间的跳跃链接以形成U-NET

    inputs = layers.Input(shape=input_shape)

    # 256x256x3 ==> 128x128x64 这些不能放在下面的for循环中，因为第一层需要接受inputs
    h = layers.Conv2D(dim, 4, strides=2, padding='same', kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(inputs)
    h = layers.BatchNormalization()(h)
    h = layers.LeakyReLU()(h)
    zs.append(h)

    # 128x128x64 ==> 64x64x128 ==> 32x32x256 ==> 16x16x512 ==> 8x8x1024
    for i in range(n_downsamplings - 1):
        d = min(dim * 2 ** (i + 1), 1024)
        h = layers.Conv2D(d, 4, strides=2, padding='same', kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(h)
        h = layers.BatchNormalization()(h)
        h = layers.LeakyReLU()(h)
        zs.append(h)

    # return Model(inputs=inputs, outputs=[h, zs], name=name)
    return Model(inputs=inputs, outputs=zs, name=name)


# print(get_G_enc().output)
# print(get_G_enc().summary())


# ==============================================================================
# =                  8x8x(1024 + n_att) ==> G_dec ==> 256x256x3                =
# ==============================================================================
def get_G_dec(zs_shape, atts_shape, dim=64, n_upsamplings=5, shortcut_layers=1, inject_layers=1, weight_decay=0.0, name='G_dec'):
    # zs = G_enc().output[1]
    # zs_shape = [z_0_shape, z_1_shape, ... ]对应着G_enc的每一层输出
    # input_shape = z_shape[1: -1] + [z_shape[-1] + atts_shape[-1]]
    inputs_1 = []
    for i in range(len(zs_shape)):
        inputs_1.append(layers.Input(shape=zs_shape[i][1:]))
    input_2 = layers.Input(shape=atts_shape)
    # inputs = inputs_1 + [input_2]
    # print(len(inputs))
    a_1 = tile_util.tile(inputs_1[-1], input_2)  # 将atts的维度扩展到enc的最后一层输出z一样（除了最后一维）
    x = layers.Concatenate()([inputs_1[-1], a_1])  # 再将z和atts进行拼接

    # 8x8x(1024 + n_att) ==> 16x16x1024 ==> 32x32x512 ==> 64x64x256 ==> 128x128x128 ==> 256x256x3
    for i in range(n_upsamplings - 1):
        d = min(dim * 2 ** (n_upsamplings - 1 - i), 1024)
        x = layers.Conv2DTranspose(d, 4, strides=2, padding='same', kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        if shortcut_layers > i:  # U-NET形式的跳跃链接
            x = layers.Concatenate()([x, inputs_1[-2 - i]])  # 将解码器第一层的输出与编码器倒数第二层的输出进行concat，并以此作为解码器第二层的输入
        if inject_layers > i:
            a_3 = tile_util.tile(x, input_2)
            x = layers.Concatenate()([x, a_3])  # x等价于: deConv(zs[-1] + a) + zs[-2] + a

    x = layers.Conv2DTranspose(3, 4, strides=2, padding='same', kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(x)
    outputs = tf.nn.tanh(x)
    return Model(inputs=inputs_1 + [input_2], outputs=outputs, name=name)


# zs = get_G_enc()
# zs_shape = []
# for z in zs.output:
#     zs_shape.append(z.shape)
# dec = get_G_dec(zs_shape, atts_shape=(10, ))
# print(dec.summary())


# ==============================================================================
# =                             256x256x3 ==> D ==> 1                          =
# =                            256x256x3 ==> C ==> n_atts                      =
# ==============================================================================
def get_D_and_C(n_atts, input_shape=(256, 256, 3), dim=64, fc_dim=1024, n_downsamplings=5, weight_decay=0.0):
    # n_atts:特征的数量，也是分类器最后一层的输出维度
    inputs = layers.Input(shape=input_shape)
    # outs = []

    # 判别器与分类器共享的卷积层
    # D/C: 256x256x3 ==> 128x128x64 ==> 64x64x128 ==> 32x32x256 ==> 16x16x512 ==> 8x8x1024 ==> 65536
    h = layers.Conv2D(dim, 4, strides=2, padding='same', kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(inputs)
    h = layers.LayerNormalization()(h)
    h = layers.LeakyReLU()(h)
    for i in range(n_downsamplings - 1):
        d = min(dim * 2 ** (i + 1), 1024)
        h = layers.Conv2D(d, 4, strides=2, padding='same', kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(h)
        h = layers.LayerNormalization()(h)
        h = layers.LeakyReLU()(h)

    h = layers.Flatten()(h)

    # 判别器拥有的FC层
    # 65536 ==> 1024 ==> 1
    h_D = layers.Dense(fc_dim, kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(h)
    # outs.append(h_D)
    h_D = layers.LayerNormalization()(h_D)
    # outs.append(h_D)
    h_D = layers.LeakyReLU()(h_D)
    # outs.append(h_D)


    outputs_D = layers.Dense(1)(h_D)
    # outs.append(outputs_D)

    # 分类器拥有的FC层
    # 65536 ==> 1024 ==> n_atts
    h_C = layers.Dense(fc_dim, kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(h)
    h_C = layers.LayerNormalization()(h_C)
    h_C = layers.LeakyReLU()(h_C)

    h_C = layers.Dense(n_atts)(h_C)
    outputs_C = tf.nn.sigmoid(h_C)

    # D = Model(inputs=inputs, outputs=outs, name='Discriminator')
    D = Model(inputs=inputs, outputs=outputs_D, name='Discriminator')
    C = Model(inputs=inputs, outputs=outputs_C, name='Classifier')
    return D, C

















