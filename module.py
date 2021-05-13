import functools

import tensorflow as tf
from tensorflow.keras import layers, Model
import tensorflow_addons as tfa

from utils import tile_util


# ==============================================================================
# =                        256x256x3 ==> G_enc ==> 8x8x1024                    =
# ==============================================================================
def G_enc(input_shape=(256, 256, 3), dim=64, n_downsamplings=5, weight_decay=0.0, name='G_enc'):
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

    return zs, Model(inputs=inputs, outputs=h, name=name)


# print(G_enc()[1].summary())


# ==============================================================================
# =     ??????       8x8x(1024 + n_att) ==> G_dec ==> 256x256x3                =
# ==============================================================================
def G_dec(zs, a, dim=64, n_upsamplings=5, shortcut_layers=1, inject_layers=1, weight_decay=0.0, name='G_dec'):
    a = tf.cast(a, tf.float32)  # 特征标签a，里面的值经过处理后变回-1/1
    z = tile_util.tile_concat(zs[-1], a)  # z是解码器的输入，z包含了：编码器的输出（zs[-1]）和对应的特征标签a
    inputs = z = layers.Input(shape=z.shape[1:])
    print(z.shape, zs[-1].shape, a.shape)

    # 8x8x(1024 + n_att) ==> 16x16x1024 ==> 32x32x512 ==> 64x64x256 ==> 128x128x128 ==> 256x256x3
    for i in range(n_upsamplings - 1):
        d = min(dim * 2 ** (n_upsamplings - 1 - i), 1024)
        z = layers.Conv2DTranspose(d, 4, strides=2, padding='same', kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(z)
        z = layers.BatchNormalization()(z)
        z = layers.ReLU()(z)
        # if shortcut_layers > i:
        #     z = tile_util.tile_concat([z, zs[-2 - i]])  # 将解码器第一层的输出与编码器倒数第二层的输出进行concat，并以此作为解码器第二层的输入
        # if inject_layers > i:
        #     z = tile_util.tile_concat(z, a)  # z等价于: deConv(zs[-1] + a) + zs[-2] + a

    z = layers.Conv2DTranspose(3, 4, strides=2, padding='same', kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(z)
    outputs = tf.nn.tanh(z)
    return Model(inputs=inputs, outputs=outputs, name=name)


# zs = [tf.zeros((64, 16, 16, 512)), tf.ones((64, 8, 8, 1024))]
# a = tf.ones((64, 10))
# print(G_dec(zs, a).summary())


# ==============================================================================
# =                                     D和C                                      =
# ==============================================================================
def D_and_C(n_atts, input_shape=(256, 256, 3), dim=64, fc_dim=1024, n_downsamplings=5, weight_decay=0.0, name='D_and_C'):
    # n_atts:特征的数量，也是分类器最后一层的输出维度
    inputs = layers.Input(shape=input_shape)

    # 判别器与分类器共享的卷积层
    h = layers.Conv2D(dim, 4, strides=2, padding='same', kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(inputs)
    h = tfa.layers.InstanceNormalization()(h)
    h = layers.LeakyReLU()(h)
    for i in range(n_downsamplings - 1):
        d = min(dim * 2 ** (i + 1), 1024)
        h = layers.Conv2D(d, 4, strides=2, padding='same', kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(h)
        h = tfa.layers.InstanceNormalization()(h)
        h = layers.LeakyReLU()(h)

    h = layers.Flatten()(h)

    # 判别器拥有的FC层
    h_D = layers.Dense(fc_dim, kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(h)
    h_D = tfa.layers.InstanceNormalization()(h_D)
    h_D = layers.LeakyReLU()(h_D)

    outputs_D = layers.Dense(1)(h_D)

    # 分类器拥有的FC层
    h_C = layers.Dense(fc_dim, kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(h)
    h_C = tfa.layers.InstanceNormalization()(h_C)
    h_C = layers.LeakyReLU()(h_C)

    h_C = layers.Dense(1)(h_C)
    outputs_C = tf.nn.sigmoid(h_C)

    D = Model(inputs=inputs, outputs=outputs_D)
    C = Model(inputs=inputs, outputs=outputs_C)
    return D, C





























