import tensorflow as tf


# ==============================================================================
# =                        将特征标签A合并到图片表示向量Z中                          =
# ==============================================================================
def tile_concat(a_list, b_list=[]):
    # `a_i` shape: (N, H, W, C_a)
    # `b_i` shape: can be (N, 1, 1, C_b) or (N, C_b)
    # a_list = [a_1, a_2, ...]
    # b_list = [b_1, b_2, ...]
    a_list = list(a_list) if isinstance(a_list, (list, tuple)) else [a_list]
    b_list = list(b_list) if isinstance(b_list, (list, tuple)) else [b_list]
    for i, b in enumerate(b_list):
        # b.shape: can be (N, 1, 1, C_b) or (N, C_b)
        b = tf.reshape(b, [-1, 1, 1, b.shape[-1]])  # b.shape: (N, 1, 1, C_b)
        # tf.tile(a, b)将a中第i维元素重复b[i]倍，若a.shape=[1, 2]，b=[2, 3]，则扩展后的a.shape=[1xb[0], 2xb[1]]=[2, 6]
        b = tf.tile(b, [1, a_list[0].shape[1], a_list[0].shape[2], 1])  # b.shape=[N, H, W, C_b]
        b_list[i] = b
    return tf.concat(a_list + b_list, axis=-1)  # 返回值的list[0].shape=[N, H, W, C_a + C_b]









