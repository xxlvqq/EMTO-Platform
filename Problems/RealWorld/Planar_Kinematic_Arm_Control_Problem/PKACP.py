# <-*--*--*--*- Reference -*--*--*--*--*->
# @title: A Bi-Objective Knowledge Transfer Framework for Evolutionary Many-Task Optimization
# @Author: Jiang, Yi and Zhan, Zhi-Hui and Tan, Kay Chen and Zhang, Jun
# @Journal: IEEE Transactions on Evolutionary Computation
# @year: 2022
# @Doi: 10.1109/TEVC.2022.3210783

# <-*--*--*--*- Reference -*--*--*--*--*->
# @title: Evolutionary Multi-Task Optimization with Adaptive Knowledge Transfer
# @Author: Xu, Hao and Qin, A. K. and Xia, Siyu
# @Journal: IEEE Transactions on Evolutionary Computation
# @year: 2021
# @Doi: 10.1109/TEVC.2021.3107435

# <-*--*--*--*- Coder -*--*--*--*--*->
# @Time: 2025/5/19 下午8:02
# @Author: wzb
# @Introduction: 多任务优化和超多任务真实世界优化问题集，Planar Kinematic Arm Control Problem(机械臂控制问题)
# @Remind: 默认评估次数为: 任务数 * 种群大小 * 进化代数

# <-*--*--*--*- Use -*--*--*--*--*->
# <Multi-task/Many-task> <Single-objective> <None>
# from Problems.RealWorld.Planar_Kinematic_Arm_Control_Problem.PKACP import PKAC
# Prob = PKAC(N=种群大小, T=任务数量, Gen=进化代数, dim=问题维度, lb=下界, ub=上界)

import numpy as np
import os
import pandas as pd
from sklearn.cluster import KMeans

from Problems.Problem import Problem


class PKACP(Problem):
    """
    PKACP类，继承自Problem类，用于定义机械臂运动学优化问题。

    :param Amax: 最大角度范围
    :param Lmax: 最大长度
    :param dim: 问题的维度（关节数量），默认值为20
    :param lb: 变量下界，默认值为0.0
    :param ub: 变量上界，默认值为1.0
    """

    def __init__(self, Amax, Lmax, dim=20, lb=0.0, ub=1.0):
        super().__init__(dim, lb, ub)
        self.Amax = Amax  # 最大角度范围
        self.Lmax = Lmax  # 最大长度

    def fnc(self, var):
        """
        计算适应度函数，调用fitness_arm计算机械臂末端执行器的适应度。

        :param var: 输入变量，形状为(N, dim)的数组，表示N个个体的关节角度
        :return:
            Objs: 适应度值，形状为(N, 1)的数组，表示每个个体的适应度
        """
        return fitness_arm(var, self.Amax, self.Lmax)


def fitness_arm(angles_var, Amax, Lmax):
    """
    计算机械臂末端执行器的适应度值，通过正向运动学计算末端位置与目标位置的欧几里得距离。

    :param angles_var: 关节角度数组，形状为(N, Dim)，N为种群大小，Dim为维度
    :param Amax: 最大角度范围
    :param Lmax: 最大长度
    :return:
        Objs: 适应度值，形状为(N, 1)的数组，表示每个个体的适应度
    """
    # angles_var: (N, Dim) 数组，N 是种群大小，Dim 是维度
    n_individuals, n_angles = angles_var.shape
    # 计算每个关节的角度范围
    angular_range = Amax / n_angles
    # 初始化每个关节的连杆长度为总长度的均分
    lengths = np.ones(n_angles) * Lmax / n_angles
    # 设置目标位置为二维平面上的点 (0.5, 0.5)
    target = np.array([0.5, 0.5])

    # 向量化计算角度，将归一化角度映射到实际角度范围
    command = (angles_var - 0.5) * angular_range * np.pi * 2  # (N, Dim)

    # 对每个个体调用正向运动学，计算末端执行器位置
    ef = np.array([fw_kinematics(command[i], lengths) for i in range(n_individuals)])  # (N, 2)

    # 计算末端执行器与目标位置的欧几里得距离
    fitness = np.sqrt(np.sum((ef - target) ** 2, axis=1))  # (N,)

    Objs = fitness[:, np.newaxis]  # (N, 1)
    return Objs


def fw_kinematics(p, lengths):
    """
    正向运动学计算，基于关节角度和连杆长度计算机械臂末端执行器的位置。

    :param p: 关节角度数组，形状为(Dim,)，表示单个个体的关节角度
    :param lengths: 连杆长度数组，形状为(Dim,)，表示每个关节的连杆长度
    :return:
        joint_xy: 末端执行器位置，形状为(2,)，表示x和y坐标
    """
    # 初始化变换矩阵为4x4单位矩阵
    mat = np.eye(4)
    p = np.append(p, 0)  # 添加一个 0 角度以匹配维度
    n_dofs = len(p)
    lengths = np.append(0, lengths)  # 添加 0 长度以匹配维度
    joint_xy = np.zeros(2)

    # 遍历每个关节，构建变换矩阵并计算末端位置
    for i in range(n_dofs):
        # 构建当前关节的变换矩阵
        m = np.array([
            [np.cos(p[i]), -np.sin(p[i]), 0, lengths[i]],  # 旋转和平移矩阵的第一行
            [np.sin(p[i]), np.cos(p[i]), 0, 0],  # 旋转和平移矩阵的第二行
            [0, 0, 1, 0],  # 保持z轴不变
            [0, 0, 0, 1]  # 齐次坐标的最后一行
        ])
        # 累积变换矩阵，更新当前的全局变换
        mat = mat @ m
        # 计算当前关节的齐次坐标
        v = mat @ np.array([0, 0, 0, 1])
        # 提取关节的二维平面位置 (x, y)
        joint_xy = v[:2]

    return joint_xy


def PKAC(N=50, T=20, Gen=200, dim=20, lb=0.0, ub=1.0):
    """
    生成PKACP问题的任务列表，基于KMeans聚类初始化任务参数。

    :param N: 种群大小，默认值为50
    :param T: 任务数量，默认值为20
    :param Gen: 迭代次数，默认值为200
    :param dim: 问题维度（关节数量），默认值为20
    :param lb: 变量下界，默认值为0.0
    :param ub: 变量上界，默认值为1.0
    :return:
        Probs: PKACP问题对象列表，长度为T，每个对象对应一个任务
    """
    # 设置任务数量和最大函数评估次数
    Problem.T = T
    Problem.maxFE = T * N * Gen
    # 获取当前文件路径，构造参数文件路径
    path = os.path.abspath(os.path.dirname(__file__))
    file = os.path.join(path, f"cvt_d{dim}_nt{Problem.T}.csv")

    # 检查参数文件是否存在，若存在则读取，否则生成并保存
    if os.path.exists(file):
        task_para = pd.read_csv(file, header=None).to_numpy()
    else:
        # 生成样本点的数量，样本点数为任务数量的50倍
        samples = 50 * Problem.T
        # 随机生成样本点，形状为(samples, 2)
        x = np.random.rand(samples, 2)
        # 使用KMeans聚类算法对样本点进行聚类，聚类数量为任务数量
        kmeans = KMeans(n_clusters=Problem.T, random_state=0)
        kmeans.fit(x)
        # 获取聚类中心作为任务参数
        task_para = kmeans.cluster_centers_
        # 将任务参数保存为CSV文件
        np.savetxt(file, task_para, delimiter=",")  # 保存为 CSV 文件

    # 初始化PKACP问题对象列表
    Probs = []
    # 遍历每个任务
    for t in range(Problem.T):
        Amax = task_para[t, 0]  # 获取当前任务的最大角度范围
        Lmax = task_para[t, 1]  # 获取当前任务的最大长度
        # 创建PKACP对象并添加到列表
        Probs.append(PKACP(Amax, Lmax, dim=dim, lb=lb, ub=ub))

    return Probs
