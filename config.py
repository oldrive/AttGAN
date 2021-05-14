# ==============================================================================
# =                                  数据设置                                   =
# ==============================================================================
ATT_ID = {'5_o_Clock_Shadow': 0, 'Arched_Eyebrows': 1, 'Attractive': 2,
          'Bags_Under_Eyes': 3, 'Bald': 4, 'Bangs': 5, 'Big_Lips': 6,
          'Big_Nose': 7, 'Black_Hair': 8, 'Blond_Hair': 9, 'Blurry': 10,
          'Brown_Hair': 11, 'Bushy_Eyebrows': 12, 'Chubby': 13,
          'Double_Chin': 14, 'Eyeglasses': 15, 'Goatee': 16,
          'Gray_Hair': 17, 'Heavy_Makeup': 18, 'High_Cheekbones': 19,
          'Male': 20, 'Mouth_Slightly_Open': 21, 'Mustache': 22,
          'Narrow_Eyes': 23, 'No_Beard': 24, 'Oval_Face': 25,
          'Pale_Skin': 26, 'Pointy_Nose': 27, 'Receding_Hairline': 28,
          'Rosy_Cheeks': 29, 'Sideburns': 30, 'Smiling': 31,
          'Straight_Hair': 32, 'Wavy_Hair': 33, 'Wearing_Earrings': 34,
          'Wearing_Hat': 35, 'Wearing_Lipstick': 36,
          'Wearing_Necklace': 37, 'Wearing_Necktie': 38, 'Young': 39}  # 40种特征
ID_ATT = {value: key for key, value in ATT_ID.items()}  # {'0': 5_o_Clock_Shadow, ...}
DEFAULT_ATT_NAMES = ['Bald', 'Bangs', 'Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Bushy_Eyebrows', 'Eyeglasses',
                     'Male', 'Mouth_Slightly_Open', 'Mustache', 'No_Beard', 'Pale_Skin', 'Young']  # 用于训练和修改的默认特征列表
LOAD_SIZE = 286  # 加载图片的大小
CROP_SIZE = 256  # 预处理之后的图片大小


# ==============================================================================
# =                                  文件设置                                   =
# ==============================================================================
IMG_DIR = 'data/img_celeba/aligned/align_size(572,572)_move(0.250,0.000)_face_factor(0.450)_jpg/data'  # 训练图片的存放目录
TRAIN_LABEL_PATH = 'data/img_celeba/train_label.txt'  # 训练数据标签的存放路径
VAL_LABEL_PATH = 'data/img_celeba/val_label.txt'  # 验证数据标签的存放路径


# ==============================================================================
# =                                  网络设置                                   =
# ==============================================================================
N_UPSAMPLINGS = 5  # G_dec中上采样（反卷积层）个数
N_DOWNSAMPLINGS = 5  # G_enc, D, C中下采样（卷积层）个数


# ==============================================================================
# =                              损失函数设置                                    =
# ==============================================================================
GP_WEIGHT = 10.0  # 梯度惩罚项的权重
G_RECONSTRUCTION_LOSS_WEIGHT = 100.0  # G（G_enc和G_dec）中重构损失的权重
G_ATTRIBUTE_LOSS_WEIGHT = 10.0  # G中特征限制损失的权重
D_ATTRIBUTE_LOSS_WEIGHT = 1.0  # D（分类器）中特征限制损失的权重


# ==============================================================================
# =                              训练设置                                       =
# ==============================================================================
N_EPOCHS = 60
EPOCH_START_DECAY = 30
BATCH_SIZE = 32
LEARNING_RATE = 2e-4
BATE_1 = 0.5

N_D = 5  # 每更新一次生成器，更新N_D次判别器


























