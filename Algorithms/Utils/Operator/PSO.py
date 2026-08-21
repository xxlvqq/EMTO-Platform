# <-*--*--*--*- Coder -*--*--*--*--*->
# @Time: 2025/7/3 14:07
# @Author: 超之皮
# @Introduction: PSO算子

import numpy as np

def pso_operator(particles, w, c1, c2, global_best, bounds=(0, 1)):
    """
    PSO算子，用于更新种群中的粒子。

    :param particles: list，包含多个粒子对象的种群
    :param w: float，惯性权重
    :param c1: float，个体学习因子
    :param c2: float，群体学习因子
    :param global_best: numpy 数组，全局最优位置
    :param bounds: tuple，搜索空间的上下界 (lower_bound, upper_bound)
    :return: list，更新后的粒子种群
    """
    lower_bound, upper_bound = bounds

    for particle in particles:

        # 更新速度
        r1 = np.random.rand(*particle.x.shape)
        r2 = np.random.rand(*particle.x.shape)
        cognitive = c1 * r1 * (particle.x - particle.x)  # 自身经验
        social = c2 * r2 * (global_best - particle.x)    # 群体经验
        particle.v = w * particle.v + cognitive + social

        # 更新位置
        particle.x = particle.x + particle.v

        # 边界控制
        particle.x = np.clip(particle.x, lower_bound, upper_bound)
        particle.v = np.clip(particle.v, -abs(upper_bound - lower_bound), abs(upper_bound - lower_bound))

    return particles