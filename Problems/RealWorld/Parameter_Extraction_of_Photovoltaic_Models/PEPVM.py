# <-*--*--*--*- Reference -*--*--*--*--*->
# @title: An Enhanced Adaptive Differential Evolution Algorithm for Parameter Extraction of Photovoltaic Models
# @Author: Shuijia Li and Qiong Gu and Wenyin Gong and Bin Ning
# @Journal: Energy Conversion and Management
# @year: 2020
# @Doi: https://doi.org/10.1016/j.enconman.2019.112443

# <-*--*--*--*- Reference -*--*--*--*--*->
# @title: Multitasking Optimization via an Adaptive Solver Multitasking Evolutionary Framework
# @Author: Yanchi Li and Wenyin Gong and Shuijia Li
# @Journal: Information Sciences
# @year: 2022
# @Doi: https://doi.org/10.1016/j.ins.2022.10.099

# <-*--*--*--*- Reference -*--*--*--*--*->
# @title: Evolutionary Competitive Multitasking Optimization via Improved Adaptive Differential Evolution
# @Author: Yanchi Li and Wenyin Gong and Shuijia Li
# @Journal: Expert Systems with Applications
# @year: 2023
# @Doi: https://doi.org/10.1016/j.eswa.2023.119550

# <-*--*--*--*- Coder -*--*--*--*--*->
# @Time: 2025/5/19 下午8:02
# @Author: wzb
# @Introduction: 多任务优化真实世界问题集，Parameter Extraction of Photovoltaic Models(光伏模型的参数提取)
# @Remind: 默认评估次数为: 1000 * 300

# <-*--*--*--*- Use -*--*--*--*--*->
# % <Multi-task> <Single-objective> <None/Competitive>
# from Problems.RealWorld.Parameter_Extraction_of_Photovoltaic_Models.PEPVM import PEPVM
# Prob = PEPVM()

import numpy as np

from Problems.Problem import Problem


class single_model(Problem):
    def __init__(self, dim=5, lb=np.array([0, 0, 0, 0, 1]), ub=np.array([1, 1e-6, 0.5, 100, 2])):
        super().__init__(dim, lb, ub)

    def fnc(self, x):
        """
        计算单二极管光伏模型的均方根误差（RMSE），用于拟合光伏阵列的I-V特性曲线
        该模型基于单二极管等效电路，考虑光生电流、饱和电流、串联电阻、并联电阻和理想因子

        :param x: numpy.ndarray, 形状为 (n, 5)，包含以下参数：
                  - x[:, 0]: I_ph，光生电流（单位：安培）
                  - x[:, 1]: I_sd，二极管饱和电流（单位：安培）
                  - x[:, 2]: R_s，串联电阻（单位：欧姆）
                  - x[:, 3]: R_sh，并联电阻（单位：欧姆）
                  - x[:, 4]: a，理想因子
        :return:
            Obj: numpy.ndarray, 均方根误差（RMSE）
        """
        # 物理常数
        q = 1.60217646e-19  # 电子电量（库仑）
        k = 1.3806503e-23  # 玻尔兹曼常数（J/K）
        T = 273.15 + 33.0  # 绝对温度（开尔文，33摄氏度）
        V_t = k * T / q  # 热电压（V）

        # 提取参数
        I_ph = x[:, 0]  # 光生电流
        I_sd = x[:, 1]  # 二极管饱和电流
        R_s = x[:, 2]  # 串联电阻
        R_sh = x[:, 3]  # 并联电阻
        a = x[:, 4]  # 理想因子

        # I-V 数据
        V_L = np.array([-0.2057, -0.1291, -0.0588, 0.0057, 0.0646, 0.1185, 0.1678, 0.2132,
                        0.2545, 0.2924, 0.3269, 0.3585, 0.3873, 0.4137, 0.4373, 0.4590,
                        0.4784, 0.4960, 0.5119, 0.5265, 0.5398, 0.5521, 0.5633, 0.5736,
                        0.5833, 0.5900])
        I_L = np.array([0.7640, 0.7620, 0.7605, 0.7605, 0.7600, 0.7590, 0.7570, 0.7570,
                        0.7555, 0.7540, 0.7505, 0.7465, 0.7385, 0.7280, 0.7065, 0.6755,
                        0.6320, 0.5730, 0.4990, 0.4130, 0.3165, 0.2120, 0.1035, -0.0100,
                        -0.1230, -0.2100])

        # 向量化计算误差
        V_Rs = V_L + I_L * R_s[:, None]  # V_L + I_L * R_s，广播到每组参数
        diode_term = I_sd[:, None] * (np.exp(V_Rs / (a[:, None] * V_t)) - 1)  # 二极管项
        shunt_term = V_Rs / R_sh[:, None]  # 分流电阻项
        y1 = I_ph[:, None] - diode_term - shunt_term - I_L  # 误差
        summ = np.sum(y1 ** 2, axis=1)  # 平方和
        Obj = np.sqrt(summ / len(V_L))  # 均方根误差

        return Obj


class double_model(Problem):
    def __init__(self, dim=7, lb=np.array([0, 0, 0, 0, 1, 0, 1]), ub=np.array([1, 1e-6, 0.5, 100, 2, 1e-6, 2])):
        super().__init__(dim, lb, ub)

    def fnc(self, var):
        """
        计算双二极管光伏模型的均方根误差（RMSE），用于拟合光伏阵列的I-V特性曲线
        该模型基于双二极管等效电路，考虑两个二极管的饱和电流和理想因子

        :param var: numpy.ndarray, 形状为 (n, 7)，包含以下参数：
                  - x[:, 0]: I_ph，光生电流（单位：安培）
                  - x[:, 1]: I_sd1，第一个二极管饱和电流（单位：安培）
                  - x[:, 2]: R_s，串联电阻（单位：欧姆）
                  - x[:, 3]: R_sh，并联电阻（单位：欧姆）
                  - x[:, 4]: a1，第一个二极管理想因子
                  - x[:, 5]: I_sd2，第二个二极管饱和电流（单位：安培）
                  - x[:, 6]: a2，第二个二极管理想因子
        :return:
            Obj: numpy.ndarray, 均方根误差（RMSE）
        """
        # 物理常数
        q = 1.60217646e-19  # 电子电量（库仑）
        k = 1.3806503e-23  # 玻尔兹曼常数（J/K）
        T = 273.15 + 33.0  # 绝对温度（开尔文，33摄氏度）
        V_t = k * T / q  # 热电压（V）

        # 提取参数
        I_ph = var[:, 0]  # 光生电流
        I_sd1 = var[:, 1]  # 第一个二极管饱和电流
        R_s = var[:, 2]  # 串联电阻
        R_sh = var[:, 3]  # 并联电阻
        a1 = var[:, 4]  # 第一个二极管理想因子
        I_sd2 = var[:, 5]  # 第二个二极管饱和电流
        a2 = var[:, 6]  # 第二个二极管理想因子

        # I-V 数据
        V_L = np.array([-0.2057, -0.1291, -0.0588, 0.0057, 0.0646, 0.1185, 0.1678, 0.2132,
                        0.2545, 0.2924, 0.3269, 0.3585, 0.3873, 0.4137, 0.4373, 0.4590,
                        0.4784, 0.4960, 0.5119, 0.5265, 0.5398, 0.5521, 0.5633, 0.5736,
                        0.5833, 0.5900])
        I_L = np.array([0.7640, 0.7620, 0.7605, 0.7605, 0.7600, 0.7590, 0.7570, 0.7570,
                        0.7555, 0.7540, 0.7505, 0.7465, 0.7385, 0.7280, 0.7065, 0.6755,
                        0.6320, 0.5730, 0.4990, 0.4130, 0.3165, 0.2120, 0.1035, -0.0100,
                        -0.1230, -0.2100])

        # 向量化计算误差
        V_Rs = V_L + I_L * R_s[:, None]  # V_L + I_L * R_s，广播到每组参数
        diode1_term = I_sd1[:, None] * (np.exp(V_Rs / (a1[:, None] * V_t)) - 1)  # 第一个二极管项
        diode2_term = I_sd2[:, None] * (np.exp(V_Rs / (a2[:, None] * V_t)) - 1)  # 第二个二极管项
        shunt_term = V_Rs / R_sh[:, None]  # 分流电阻项
        y1 = I_ph[:, None] - diode1_term - diode2_term - shunt_term - I_L  # 误差
        summ = np.sum(y1 ** 2, axis=1)  # 平方和
        Obj = np.sqrt(summ / len(V_L))  # 均方根误差

        return Obj


class pv_model(Problem):
    def __init__(self, dim=5, lb=np.array([0, 0, 0, 0, 1]), ub=np.array([2, 5e-5, 2, 2000, 50])):
        super().__init__(dim, lb, ub)

    def fnc(self, var):
        """
        计算单二极管光伏模型的均方根误差（RMSE），考虑光伏模块串联单元数，适用于不同温度条件
        该模型用于拟合光伏阵列在45摄氏度下的I-V特性曲线

        :param var: numpy.ndarray, 形状为 (n, 5)，包含以下参数：
                  - x[:, 0]: I_ph，光生电流（单位：安培）
                  - x[:, 1]: I_sd，二极管饱和电流（单位：安培）
                  - x[:, 2]: R_s，串联电阻（单位：欧姆）
                  - x[:, 3]: R_sh，并联电阻（单位：欧姆）
                  - x[:, 4]: a，理想因子
        :return:
            Obj: numpy.ndarray, 均方根误差（RMSE）
        """
        # 物理常数
        q = 1.60217646e-19  # 电子电量（库仑）
        k = 1.3806503e-23  # 玻尔兹曼常数（J/K）
        T = 273.15 + 45.0  # 绝对温度（开尔文，45摄氏度）
        V_t = k * T / q  # 热电压（V）
        Ns = 1  # 光伏模块串联单元数

        # 提取参数
        I_ph = var[:, 0]  # 光生电流
        I_sd = var[:, 1]  # 二极管饱和电流
        R_s = var[:, 2]  # 串联电阻
        R_sh = var[:, 3]  # 并联电阻
        a = var[:, 4]  # 理想因子

        # I-V 数据
        V_L = np.array([0.1248, 1.8093, 3.3511, 4.7622, 6.0538, 7.2364, 8.3189, 9.3097,
                        10.2163, 11.0449, 11.8018, 12.4929, 13.1231, 13.6983, 14.2221,
                        14.6995, 15.1346, 15.5311, 15.8929, 16.2229, 16.5241, 16.7987,
                        17.0499, 17.2793, 17.4885])
        I_L = np.array([1.0315, 1.0300, 1.0260, 1.0220, 1.0180, 1.0155, 1.0140, 1.0100,
                        1.0035, 0.9880, 0.9630, 0.9255, 0.8725, 0.8075, 0.7265, 0.6345,
                        0.5345, 0.4275, 0.3185, 0.2085, 0.1010, -0.0080, -0.1110, -0.2090,
                        -0.3030])

        # 向量化计算误差
        V_Rs = V_L + I_L * R_s[:, None]  # V_L + I_L * R_s，广播到每组参数
        diode_term = I_sd[:, None] * (np.exp(V_Rs / (a[:, None] * Ns * V_t)) - 1)  # 二极管项
        shunt_term = V_Rs / R_sh[:, None]  # 分流电阻项
        y1 = I_ph[:, None] - diode_term - shunt_term - I_L  # 误差
        summ = np.sum(y1 ** 2, axis=1)  # 平方和
        Obj = np.sqrt(summ / len(V_L))  # 均方根误差

        return Obj


def PEPVM():
    """
    生成光伏模型参数提取问题的三个不同实例列表(分别对应单二极管、双二极管和光伏模块模型)

    :return:
        Probs: 优化问题实例列表，每个实例配置了特定的函数
    """
    # 设置任务总数
    Problem.T = 3
    # 设置最大函数评估次数（每任务默认100个个体，进化1000代）
    Problem.maxFE = 1000 * 100 * Problem.T

    # 初始化问题实例列表
    Probs = [
        single_model(dim=5, lb=np.array([0, 0, 0, 0, 1]), ub=np.array([1, 1e-6, 0.5, 100, 2])),
        double_model(dim=7, lb=np.array([0, 0, 0, 0, 1, 0, 1]), ub=np.array([1, 1e-6, 0.5, 100, 2, 1e-6, 2])),
        pv_model(dim=5, lb=np.array([0, 0, 0, 0, 1]), ub=np.array([2, 5e-5, 2, 2000, 50]))
    ]

    return Probs
