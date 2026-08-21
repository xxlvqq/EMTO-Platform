# <-*--*--*--*- Coder -*--*--*--*--*->
# @Time: 2025/4/14 下午8:02
# @Author: wzb
# @Introduction: 个体基类，用于表示优化问题中的个体

import numpy as np


class Individual:
    def __init__(self, rnvec=None, obj=None):
        """
        初始化个体。

        :param rnvec: 个体的基因型，默认为空的浮点型NumPy数组
        :param obj: 个体的适应度值，默认为None
        """
        self.rnvec = rnvec if rnvec is not None else np.empty(0, dtype=float)  # 基因型，初始化为空NumPy数组
        self.obj = obj  # 适应度，初始化为None
