# 对文件，目录，路径进行各种操作


import os
import glob as _glob


def mkdir(paths):
    # 创建指定目录
    if not isinstance(paths, (list, tuple)):
        paths = [paths]
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)


def glob(dir, pats, recursive=False):
    # 返回匹配到的所有文件的相对路径: ['data/1.jpg', 'data/2.jpg', ...]
    pats = pats if isinstance(pats, (list, tuple)) else [pats]
    matches = []
    for pat in pats:
        matches += _glob.glob(os.path.join(dir, pat), recursive=recursive)
    return matches