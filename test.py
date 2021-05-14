import tensorflow as tf
import numpy as np
from utils import path_util
import config
import data
from tensorflow.keras import layers, Model, Sequential

# label_path = 'data/img_celeba/train_label.txt'
# img_names = np.genfromtxt(label_path, dtype=str, usecols=0)
# img_dir = 'data/img_celeba/aligned/align_size(572,572)_move(0.250,0.000)_face_factor(0.450)_jpg/data'
# # print(path_util.glob(img_dir, img_names.tolist()))
# labels = np.genfromtxt(label_path, dtype=int, usecols=range(1, 41))
# att_names = ['Bald', 'Bangs', 'Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Bushy_Eyebrows', 'Eyeglasses',
#                      'Male', 'Mouth_Slightly_Open', 'Mustache', 'No_Beard', 'Pale_Skin', 'Young']
#
# datasets, img_shape, len_datasets = data.make_celeba_dataset(img_dir, label_path, att_names, batch_size=32, training=False)
# for dataset, label in datasets.take(1):
#     print(dataset[0].shape, label[0])


# def tile_concat(a_list, b_list=[]):
#     # tile all elements of `b_list` and then concat `a_list + b_list` along the channel axis
#     # `a` shape: (N, H, W, C_a)
#     # `b` shape: can be (N, 1, 1, C_b) or (N, C_b)
#     a_list = list(a_list) if isinstance(a_list, (list, tuple)) else [a_list]
#     b_list = list(b_list) if isinstance(b_list, (list, tuple)) else [b_list]
#     for i, b in enumerate(b_list):
#         b = tf.reshape(b, [-1, 1, 1, b.shape[-1]])
#         b = tf.tile(b, [1, a_list[0].shape[1], a_list[0].shape[2], 1])
#         b_list[i] = b
#     # print(a_list[0].shape, tf.concat(a_list + b_list, axis=-1).shape)
#     return tf.concat(a_list + b_list, axis=-1)


# a = np.arange(24).reshape((2, 2, 2, 3))
# a = tf.constant(a)
# c = np.arange(16).reshape((2, 2, 2, 2))
# c = tf.constant(c)
# b = np.arange(20).reshape((2, 10))
# b = tf.constant(b)
# print('a=================')
# print(a)
# print('c=================')
# print(c)
# print('a+c=================')
# print(tile_concat([a, c]))
# shape = (1, 2, 3, 4)
# print(shape[1:])


# inputs = layers.Input(shape=[2, ])
# h = layers.Dense(3)(inputs)
# shared_layers = Model(inputs=inputs, outputs=h)
# print("=================shared===============")
# print(shared_layers.trainable_weights)
# # D = Sequential()
# # D.add(shared_layers)
# # D.add(layers.Dense(2))
# # C = Sequential()
# # C.add(shared_layers)
# # C.add(layers.Dense(1))
# h_D = layers.Dense(2)(h)
# D = Model(inputs=inputs, outputs=h_D)
# h_C = layers.Dense(2)(h)
# C = Model(inputs=inputs, outputs=h_C)
# print("=================D===============")
# print(D.trainable_weights)
# print("=================C===============")
# print(C.trainable_weights)
# print('=================change===============')
# shared_layers.trainable_weights[0].assign(tf.clip_by_value(shared_layers.trainable_weights[0], -0.01, 0.01))
# print("=================shared===============")
# print(shared_layers.trainable_weights)
# print("=================D===============")
# print(D.trainable_weights)
# print("=================C===============")
# print(C.trainable_weights)

# s1 = [64, 2, 2, 3]
# s2 = [64, 10]
# s = s1[1: -1] + [s1[-1] + s2[-1]]
# print(s)
# layers.concatenate()


a = [1, 2, 3]
print(a + [4])













