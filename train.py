import tensorflow as tf
tf.config.experimental.set_memory_growth(tf.config.list_physical_devices("GPU")[0], True)  # 设置GPU显存用量按需使用, 需要紧跟在import tf后面设置GPU显存，不然后面导入的包可能会实例化模型，造成显存分配失败

import os
import tqdm

import config, dataset, loss, module
from utils import path_util


# ==============================================================================
# =                               0.   output_dir                              =
# ==============================================================================
output_dir = os.path.join('output', 'celeba_attGAN')
path_util.mkdir(output_dir)


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
# print(G_dec.input)
for imgs_test, atts_test in train_dataset.take(1):  # 确保G_enc有五个输出
    zs_test = G_enc(imgs_test)

    print(len(imgs_test))

    G_dec(zs_test + [atts_test])  # 多输入的模型，在输入数据时需要传入列表进去
    # print(len(G_enc(imgs_test)))



# ==============================================================================
# =                               3.   train_step                              =
# ==============================================================================
@tf.function
def train_step_G():
    pass


@tf.function
def train_step_D(real_x, real_atts):
    '''

    :param real_x:  训练集中的真实图片
    :param real_atts:  真实图片中具有的特征，值为0/1
    :return:
    '''
    with tf.GradientTape() as tape:
        fake_atts = tf.random.shuffle(real_atts)  # 为了进行人脸编辑，先准备每一张图片的atts，这里将图片对应的atts打乱了，以便产生具有新特征的图片

        real_z = G_enc(real_x)
        fake_x = G_dec()











