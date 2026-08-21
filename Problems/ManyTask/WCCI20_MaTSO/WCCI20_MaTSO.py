# <-*--*--*--*- Coder -*--*--*--*--*->
# @Time: 2025/5/18 下午3:30
# @Author: zsy、wzb
# @Introduction: CEC22(WCCI20/22) 超多任务优化问题集
# @Remind: 默认评估次数为5000000，种群大小为100

# <-*--*--*--*- Use -*--*--*--*--*->
# <Many-task> <Single-objective> <None>
# from Problems.ManyTask.WCCI20_MaTSO.WCCI20_MaTSO import WCCI20_MaTSO
# Prob = WCCI20_MaTSO(ProbNum=问题编号)

import numpy as np
import os
from Problems.Base import *
from Problems.Problem import Problem

# 按顺序定义的基准函数类列表
Funcs = [
    Sphere,
    Rosenbrock,
    Ackley,
    Rastrigin,
    Griewank,
    Weierstrass,
    Schwefel
]

# 每个函数的默认参数，与 functions 列表顺序对应
default_params = [
    {"dim": 50, "lb": -100, "ub": 100},  # Sphere 函数
    {"dim": 50, "lb": -50, "ub": 50},  # Rosenbrock 函数
    {"dim": 50, "lb": -50, "ub": 50},  # Ackley 函数
    {"dim": 50, "lb": -50, "ub": 50},  # Rastrigin 函数
    {"dim": 50, "lb": -100, "ub": 100},  # Griewank 函数
    {"dim": 50, "lb": -0.5, "ub": 0.5},  # Weierstrass 函数
    {"dim": 50, "lb": -500, "ub": 500}  # Schwefel 函数
]

# 每个问题编号对应的函数索引列表
FuncNums = [
    [0], [1], [3],
    [0, 1, 2], [3, 4, 5], [1, 4, 6], [2, 3, 5],
    [1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 6], [2, 3, 4, 5, 6]
]


def loadfile(ProbNum, FuncNum):
    """
    加载特定问题和函数索引的变换矩阵和偏置向量。

    :param ProbNum: 整数，问题编号（1-10）
    :param FuncNum: 整数，函数编号（1-50）
    :return:
        matrix: numpy 数组，从文件中加载的变换矩阵
        bias: numpy 数组，从文件中加载的偏置向量
    """
    # 获取当前文件所在目录的绝对路径
    path = os.path.abspath(os.path.dirname(__file__))
    # 构建基准问题文件夹路径
    file = os.path.join(path, 'Tasks', 'benchmark_' + str(ProbNum))
    # 加载变换矩阵
    matrix = np.loadtxt(os.path.join(file, 'matrix_' + str(FuncNum)), dtype=float, delimiter=None)
    # 加载偏置向量
    bias = np.loadtxt(os.path.join(file, 'bias_' + str(FuncNum)), dtype=float, delimiter=None)

    return matrix, bias


def WCCI20_MaTSO(ProbNum=1):
    """
    为 WCCI20 超多任务优化基准生成优化问题实例列表。

    :param ProbNum: 整数，问题编号，用于选择函数集合（1-10）
    :return:
        Probs: 优化问题实例列表，每个实例配置了特定的函数、矩阵、偏置和参数
    """
    # 处理异常输入
    if not isinstance(ProbNum, int) or not (1 <= ProbNum <= 10):
        raise ValueError("ProbNum must be an integer between 1 and 10.")

    # 设置任务总数
    Problem.T = 50
    # 设置最大函数评估次数（每任务默认100个个体，进化1000代）
    Problem.maxFE = 1000 * 100 * Problem.T
    # 初始化问题实例列表
    Probs = []
    # 获取指定问题编号的函数索引列表
    FuncList = FuncNums[ProbNum - 1]

    # 创建 T 个问题实例（根据函数索引循环创建）
    for i in range(Problem.T):
        # 为当前任务加载变换矩阵和偏置
        matrix, bias = loadfile(ProbNum, i + 1)
        # 循环选择函数索引
        func_idx = FuncList[i % len(FuncList)]
        # 获取对应的函数类
        func_class = Funcs[func_idx]
        # 获取函数的默认参数
        params = default_params[func_idx]
        # 创建函数的新实例，配置加载的矩阵、偏置和默认参数
        func_instance = func_class(
            M=matrix,
            opt=bias,
            dim=params["dim"],
            lb=params["lb"],
            ub=params["ub"],
            g=0
        )
        # 将实例添加到问题列表
        Probs.append(func_instance)

    return Probs
