# <-*--*--*--*- Coder -*--*--*--*--*->
# @Time: 2025/4/14 下午8:02
# @Author: wzb
# @Introduction: 多种群选择算子
# @Remind: 同索引的父代与子代构成锦标赛池

import numpy as np

from Problems.Problem import Problem


def Selection_Elit(population, offspring, num=Problem.N):
    """
    根据因子成本，使用精英保留策略更新新种群，并返回父代存活标志。

    :param population: list，父代个体列表
    :param offspring: list，子代个体列表
    :param num: int，保留的个体数量，默认为 Problem.N
    :return:
        new_population: list，精英保留后的个体列表
        survive: numpy.array，标志每个父代个体是否存活（True 表示存活，False 表示淘汰）
    """
    # 合并父代与子代
    combined = population + offspring
    n_pop = len(population)

    # 提取适应度值
    costs = np.fromiter((ind.obj for ind in combined), dtype=float)

    # 获取按适应度升序排序的前 num 个索引
    best_idx = np.argsort(costs)[:num]

    # 构建新种群
    new_population = [combined[i] for i in best_idx]

    # 计算父代存活标志（索引 < n_pop 表示来自父代）
    survive = np.zeros(n_pop, dtype=bool)
    survive[best_idx[best_idx < n_pop]] = True

    return new_population, survive


def Selection_Tournament_Aligned(population, offspring, num=Problem.N):
    """
    使用锦标赛选择更新新种群（同索引的父代与子代进行比较），返回父代存活标志。

    :param population: list，父代个体列表
    :param offspring: list，子代个体列表
    :param num: int，保留的个体数量，默认为 Problem.N
    :return:
        population: list，锦标赛选择后的个体列表
        survive: numpy.array，标志每个父代个体是否存活（True 表示父代存活，False 表示被子代替换）
    """
    num = min(num, len(population), len(offspring))
    # 提取父代和子代的适应度值
    pop_fitness = np.array([ind.obj for ind in population], dtype=float)
    off_fitness = np.array([ind.obj for ind in offspring], dtype=float)

    # 比较父代和子代适应度，父代更优（值更小）或相等时存活
    survive = pop_fitness[:num] <= off_fitness[:num]

    # 更新种群：选择父代或子代
    population = [offspring[i] if not survive[i] else population[i] for i in range(num)]

    return population, survive


def Selection_Tournament_Global(population, offspring, num=Problem.N, k=5):
    """
    使用锦标赛选择更新新种群（不要求对齐索引），从父代和子代中选择最优个体，获胜个体不重复参与锦标赛。

    :param population: list，父代个体列表
    :param offspring: list，子代个体列表
    :param num: int，保留的个体数量，默认为 Problem.N
    :param k: int，锦标赛池大小，默认为 5
    :return:
        new_population: list，更新后的种群
        survive: numpy.array，标志每个父代个体是否存活（True 表示父代存活，False 表示淘汰）
    """
    # 合并父代和子代
    combined = population + offspring
    n_total = len(combined)

    # 提取适应度值
    fitness = np.fromiter((ind.obj for ind in combined), dtype=float)

    # 初始化存活标志
    survive = np.zeros(n_total, dtype=bool)

    # 跟踪可用个体
    available_mask = np.ones(n_total, dtype=bool)
    # 初始化选择的个体索引
    selected_indices = np.zeros(min(num, n_total), dtype=int)

    # 进行锦标赛选择
    for i in range(min(num, n_total)):
        # 获取可用个体的索引
        available_indices = np.where(available_mask)[0]
        # 获取锦标赛池大小
        k_actual = min(k, len(available_indices))
        if k_actual == 0:
            break

        # 随机选择 k_actual 个个体
        tournament_indices = np.random.choice(available_indices, size=k_actual, replace=False)
        tournament_fitness = fitness[tournament_indices]

        # 选择适应度最优的个体
        winner_local_idx = np.argmin(tournament_fitness)
        # 获取获胜个体在原始列表中的索引
        winner_idx = tournament_indices[winner_local_idx]
        # 将获胜个体的索引存入选中列表
        selected_indices[i] = winner_idx
        # 标记获胜个体为存活
        survive[winner_idx] = True
        # 加入已选择的个体后，将其从可用个体中移除
        available_mask[winner_idx] = False

    # 构建新种群
    new_population = [combined[i] for i in selected_indices]

    return new_population, survive[:len(population)]
