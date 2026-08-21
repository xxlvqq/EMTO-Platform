# <-*--*--*--*- Coder -*--*--*--*--*->
# @Time: 2025/4/14 下午9:12
# @Author: wzb
# @Introduction: 选择算子

import numpy as np


def RouletteWheelSelection(values, num=1):
    """
    基于轮盘赌选择算法选择多个个体的索引。

    :param values: 所有个体的适应度值列表或数组
    :param num: 需要选择的个体数量
    :return: 选中的个体索引
    """
    values = np.asarray(values, dtype=np.float64)
    total_fitness = np.sum(values)

    # 生成num个随机点
    rand_points = np.random.uniform(low=0, high=total_fitness, size=num)

    # 计算累积和
    cumsum_fitness = np.cumsum(values)

    # 使用search-sorted进行向量化查找
    selected_indices = np.searchsorted(cumsum_fitness, rand_points)

    return selected_indices


def StochasticUniversalSampling(values, num=1):
    """
    基于随机遍历选择算法选择个体的索引。

    :param values: 所有个体的取值（列表或数组）
    :param num: 选择个体数（整数）
    :return: 选中的个体索引
    """
    values = np.asarray(values, dtype=np.float64)
    total_fitness = np.sum(values)

    # 计算步长和随机起始点
    step = total_fitness / num
    start_point = np.random.uniform(0, step)

    # 生成等间隔的选择点
    points = start_point + np.arange(num) * step

    # 计算累积和并查找索引
    cumsum_fitness = np.cumsum(values)
    selected_indices = np.searchsorted(cumsum_fitness, points)

    return selected_indices


def TournamentSelection(values, num=1, k=5):
    """
    基于锦标赛选择策略选择个体的索引（值较大的）。

    :param values: 所有个体的取值（列表或数组）
    :param num: 选择个体数（整数）
    :param k: 锦标赛池大小（整数）
    :return: 选中的个体索引
    """
    values = np.asarray(values, dtype=np.float64)
    n = len(values)

    # 一次性生成所有锦标赛的随机索引
    tournament_indices = np.random.choice(n, size=(num, k), replace=False)

    # 获取每个锦标赛的适应度值
    tournament_fitness = values[tournament_indices]

    # 选择每个锦标赛中最大适应度的索引
    winner_indices = tournament_indices[np.arange(num), np.argmax(tournament_fitness, axis=1)]

    return winner_indices
