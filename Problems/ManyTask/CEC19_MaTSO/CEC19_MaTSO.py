# <-*--*--*--*- Coder -*--*--*--*--*->
# @Time: 2025/5/18 下午3:30
# @Author: zsy、wzb
# @Introduction: CEC19 超多任务优化问题集
# @Remind: 默认评估次数为5000000，种群大小为100

# <-*--*--*--*- Use -*--*--*--*--*->
# <Many-task> <Single-objective> <None>
# from Problems.ManyTask.CEC19_MaTSO.CEC19_MaTSO import CEC19_MaTSO
# Prob = CEC19_MaTSO(ProbNum=问题编号)

import numpy as np
import scipy.io as sio
import os
from Problems.Base import *
from Problems.Problem import Problem

# 按顺序定义的基准函数类列表
Funcs = [
    Rosenbrock,  # Rosenbrock 函数
    Ackley,  # Ackley 函数
    Rastrigin,  # Rastrigin 函数
    Griewank,  # Griewank 函数
    Weierstrass,  # Weierstrass 函数
    Schwefel  # Schwefel 函数
]

# 每个函数的默认参数，与 Funcs 列表顺序对应
default_params = [
    {"dim": 50, "lb": -50, "ub": 50},  # Rosenbrock 函数
    {"dim": 50, "lb": -50, "ub": 50},  # Ackley 函数
    {"dim": 50, "lb": -50, "ub": 50},  # Rastrigin 函数
    {"dim": 50, "lb": -100, "ub": 100},  # Griewank 函数
    {"dim": 50, "lb": -0.5, "ub": 0.5},  # Weierstrass 函数
    {"dim": 50, "lb": -500, "ub": 500}  # Schwefel 函数
]


def mat2python(filename):
    """
    从 MATLAB 文件加载数据。

    :param filename: 字符串，MATLAB 文件名（不含 .mat 扩展名）
    :return:
        data: 从 MATLAB 文件中加载的数据
    """
    # 获取当前文件所在目录的绝对路径
    path = os.path.abspath(os.path.dirname(__file__))
    # 构建 MATLAB 文件路径
    file = os.path.join(path, 'Tasks', filename + '.mat')
    # 加载 MATLAB 文件
    data = sio.loadmat(str(file))
    # 返回文件中与文件名对应的数据
    return data[filename]


def CEC19_MaTSO(ProbNum=1):
    """
    为 CEC19 超多任务优化基准生成优化问题实例列表。

    :param ProbNum: 整数，问题编号，用于选择对应的基准函数（1-6）
    :return:
        Probs: 优化问题实例列表，每个实例配置了特定的函数、旋转矩阵、偏置向量和参数
    """
    # 处理异常输入
    if not isinstance(ProbNum, int) or not (1 <= ProbNum <= 6):
        raise ValueError("ProbNum must be an integer between 1 and 6.")

    # 设置任务总数
    Problem.T = 50
    # 设置最大函数评估次数（每任务默认100个个体，进化1000代）
    Problem.maxFE = 1000 * 100 * Problem.T
    # 初始化问题实例列表
    Probs = []

    # 获取当前问题的函数类
    CurFunc = Funcs[ProbNum - 1]
    # 构建旋转矩阵和偏置向量的文件名
    RotateMat = 'RotationTask' + str(ProbNum)
    BiasVec = 'GoTask' + str(ProbNum)
    # 加载旋转矩阵和偏置向量
    RotateMat = mat2python(RotateMat)[0]
    BiasVec = mat2python(BiasVec)

    # 获取当前函数的默认参数
    params = default_params[ProbNum - 1]

    # 创建 T 个问题实例
    for t in range(Problem.T):
        # 创建函数实例，配置旋转矩阵、偏置向量和默认参数
        Task = CurFunc(
            M=RotateMat[t],  # 旋转矩阵
            opt=BiasVec[t],  # 偏置向量
            dim=params["dim"],  # 维度
            lb=params["lb"],  # 下界
            ub=params["ub"],  # 上界
            g=0  # 偏移量
        )
        # 将实例添加到问题列表
        Probs.append(Task)

    return Probs
