import tensorflow as tf
import numpy as np
from utils import path_util
import config
import data

label_path = 'data/img_celeba/train_label.txt'
img_names = np.genfromtxt(label_path, dtype=str, usecols=0)
img_dir = 'data/img_celeba/aligned/align_size(572,572)_move(0.250,0.000)_face_factor(0.450)_jpg/data'
# print(path_util.glob(img_dir, img_names.tolist()))
labels = np.genfromtxt(label_path, dtype=int, usecols=range(1, 41))
att_names = ['Bald', 'Bangs', 'Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Bushy_Eyebrows', 'Eyeglasses',
                     'Male', 'Mouth_Slightly_Open', 'Mustache', 'No_Beard', 'Pale_Skin', 'Young']

datasets, img_shape, len_datasets = data.make_celeba_dataset(img_dir, label_path, att_names, batch_size=32, training=False)
for dataset, label in datasets.take(1):
    print(dataset[0].shape, label[0])





