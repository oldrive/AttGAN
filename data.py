import tensorflow as tf
import numpy as np
import os

import config
from utils import path_util, dataset_util


# ==============================================================================
# =                                 datasets                                   =
# ==============================================================================
def make_celeba_dataset(img_dir,
                        label_path,
                        att_names,
                        batch_size,
                        load_size=286,
                        crop_size=256,
                        training=True,
                        drop_remainder=True,
                        shuffle=True,
                        repeat=1):
    img_names = np.genfromtxt(label_path, dtype=str, usecols=0)  # 从标签文件中获取所有图片的文件名字：['01.jpg', '02.jpg', ...]
    # img_paths = path_util.glob(img_dir, img_names.tolist())  # 所有图片的相对路径
    img_paths = np.array([os.path.join(img_dir, img_name) for img_name in img_names])
    labels = np.genfromtxt(label_path, dtype=int, usecols=range(1, 41))  # 含有全部40个特征的标签
    labels = labels[:, np.array([config.ATT_ID[att_name] for att_name in att_names])]  # 只有想要进行操作的特征对应的标签（少于等于40个）

    if shuffle:
        idx = np.random.permutation(len(img_paths))
        img_paths = img_paths[idx]
        labels = labels[idx]

    if training:
        # map函数：对每张图片和标签进行操作，使得图片值处于[0, 1]，标签值为0/1
        @tf.function
        def map_fn_(img, label):
            img = tf.image.resize(img, [load_size, load_size])
            # img = tl.random_rotate(img, 5)
            img = tf.image.random_flip_left_right(img)
            img = tf.image.random_crop(img, [crop_size, crop_size, 3])
            # img = tl.color_jitter(img, 25, 0.2, 0.2, 0.1)
            # img = tl.random_grayscale(img, p=0.3)
            img = tf.clip_by_value(img, 0, 255) / 127.5 - 1
            label = (label + 1) // 2
            return img, label
    else:
        @tf.function
        def map_fn_(img, label):
            img = tf.image.resize(img, [load_size, load_size])
            img = tf.image.crop_to_bounding_box(img, (tf.shape(img)[-3] - crop_size) // 2, (tf.shape(img)[-2] - crop_size) // 2, crop_size, crop_size)
            img = tf.clip_by_value(img, 0, 255) / 127.5 - 1
            label = (label + 1) // 2
            return img, label

    dataset = dataset_util.disk_image_batch_dataset(img_paths,
                                                    batch_size,
                                                    labels=labels,
                                                    drop_remainder=drop_remainder,
                                                    map_fn=map_fn_,
                                                    shuffle=shuffle,
                                                    repeat=repeat)

    img_shape = [crop_size, crop_size, 3]

    if drop_remainder:
        len_dataset = len(img_paths) // batch_size
    else:
        len_dataset = int(np.ceil(len(img_paths) / batch_size))

    return dataset, img_shape, len_dataset


# ==============================================================================
# =                                 特征标签的冲突处理                            =
# ==============================================================================
def check_attribute_conflict(att_batch, att_name, att_names):
    def _set(att, value, att_name):
        if att_name in att_names:
            att[att_names.index(att_name)] = value

    idx = att_names.index(att_name)

    for att in att_batch:
        if att_name in ['Bald', 'Receding_Hairline'] and att[idx] == 1:  # 特征“秃子”，“高发际线”和“刘海”有冲突
            _set(att, 0, 'Bangs')
        elif att_name == 'Bangs' and att[idx] == 1:  # 有刘海就不能为秃子或者高发际线
            _set(att, 0, 'Bald')
            _set(att, 0, 'Receding_Hairline')
        elif att_name in ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair'] and att[idx] == 1:  # 头发颜色之间有冲突
            for n in ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair']:
                if n != att_name:
                    _set(att, 0, n)
        elif att_name in ['Straight_Hair', 'Wavy_Hair'] and att[idx] == 1:  # 直发与卷发之间有冲突
            for n in ['Straight_Hair', 'Wavy_Hair']:
                if n != att_name:
                    _set(att, 0, n)
        elif att_name in ['Mustache', 'No_Beard'] and att[idx] == 1:  # 有胡子与没胡子之间有冲突
            for n in ['Mustache', 'No_Beard']:
                if n != att_name:
                    _set(att, 0, n)

    return att_batch

















