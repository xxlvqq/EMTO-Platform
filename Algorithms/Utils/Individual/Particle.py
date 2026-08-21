# <-*--*--*--*- Coder -*--*--*--*--*->
# @Time: 2025/7/3 14:40
# @Author: 超之皮
# @Introduction: 粒子类
import numpy as np


class Particle:
    def __init__(self, x=np.empty(0, dtype=float), v=np.empty(0, dtype=float), obj=None):
        """
        粒子属性
        :param x: 粒子的位置，默认为空的浮点型NumPy数组
        :param v: 粒子的速度
        :param obj: 个体的适应度值，默认为None
        """
        self.x = x
        self.v = v
        self.obj = obj
        # 粒子的最佳位置
        self.best_x = None