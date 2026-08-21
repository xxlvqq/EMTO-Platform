# <-*--*--*--*- Coder -*--*--*--*--*->
# @Time: 2025/4/14 下午8:02
# @Author: wcl、wzb
# @Introduction: CEC22(WCCI20/22) 多任务优化问题集
# @Remind: 默认评估次数为200000，种群大小为100

# <-*--*--*--*- Use -*--*--*--*--*->
# <Multi-task> <Single-objective> <None>
# from Problems.MultiTask.WCCI20_MTSO.WCCI20_MTSO import WCCI20_MTSO
# Prob = Benchmark1()/Benchmark2()/Benchmark3()/Benchmark4()/Benchmark5()/Benchmark6()/Benchmark7()/Benchmark8()/Benchmark9()/Benchmark10()

import copy
import os
from abc import abstractmethod

import numpy as np

from Problems.Problem import Problem


class Task(Problem):
    def __init__(self, lb=0, ub=1, dim=50, M=None, opt=None, shuffle=None, sh_rate=1.0):
        """
        初始化任务类，设置边界、维度、旋转矩阵、偏置和缩放率。

        :param lb: 下界（标量或数组），默认 0
        :param ub: 上界（标量或数组），默认 1
        :param dim: 问题的维度，默认 50
        :param M: 旋转矩阵（dim x dim），默认单位矩阵
        :param opt: 偏置/中心向量（dim,），默认零向量
        :param shuffle: 可选的打乱参数，默认 None
        :param sh_rate: 缩放率，默认 1.0
        """
        super().__init__(dim, lb, ub)

        # 如果提供了偏置，则使用提供的偏置，否则初始化为零向量
        self.center = np.zeros(self.dim) if opt is None else np.asarray(opt, dtype=float)
        # 如果提供了系数矩阵，则使用提供的系数矩阵，否则初始化为单位矩阵
        self.M = np.eye(self.dim) if M is None else M
        # 设置随机打乱序列
        self.shuffle = shuffle
        # 设置缩放率
        self.sh_rate = sh_rate

    def decode(self, var):
        """
        预处理变量：解码、平移、旋转、缩放，将0-1之间的值转换为真实值

        :param var: 输入矩阵，值在 [0,1]，形状为 (n, dim)，n 为种群大小
        :return: 转换后的变量，形状为 (n, dim)
        """
        # 确保 X 是二维矩阵
        var = np.atleast_2d(var)  # 形状为 (n, dim)
        # 将输入X从[0, 1]范围映射到[lb, ub]范围
        var = self.lb + (self.ub - self.lb) * var
        # 截取前dim个元素
        var = var[:, :self.dim]
        # 减去中心偏置（偏置）
        var = var - self.center
        # 乘以缩放率（放缩）
        var = var * self.sh_rate
        # 旋转：对每行应用矩阵 M，M 形状为 (dim, dim)
        var = np.array([np.dot(self.M, row) for row in var])

        return var

    @abstractmethod
    def fnc(self, var):
        """
        抽象方法，子类实现具体算法流程

        :param var: 种群基因型矩阵
        :return: Objs: 适应度值
        """
        raise NotImplementedError("Subclasses should implement this method")


# 函数子类
class Ellips(Task):
    def __init__(self, lb=-100, ub=100, dim=50, M=None, opt=None, shuffle=None, sh_rate=1.0):
        super().__init__(lb, ub, dim, M, opt, shuffle, sh_rate)

    def fnc(self, var, tag=False):
        # 变量预处理
        if not tag:
            var = self.decode(var)
        # 计算权重
        weights = 10 ** (6 * np.arange(self.dim) / (self.dim - 1))  # 形状为 (dim,)
        # 计算每行的加权平方和
        obj = np.sum(weights * var ** 2, axis=1)

        return obj  # 形状为 (n,)


class Discus(Task):
    def __init__(self, lb=-100, ub=100, dim=50, M=None, opt=None, shuffle=None, sh_rate=1.0):
        super().__init__(lb, ub, dim, M, opt, shuffle, sh_rate)

    def fnc(self, var, tag=False):
        # 变量预处理
        if not tag:
            var = self.decode(var)
        # 计算目标值：第一维度平方乘以 10^6，其余维度平方和
        obj = (var[:, 0] ** 2) * (10 ** 6) + np.sum(var[:, 1:] ** 2, axis=1)

        return obj  # 形状为 (n,)


class Rosenbrock(Task):
    def __init__(self, lb=-100, ub=100, dim=50, M=None, opt=None, shuffle=None, sh_rate=2.048 / 100.0):
        super().__init__(lb, ub, dim, M, opt, shuffle, sh_rate)

    def fnc(self, var, tag=False):
        # 变量预处理
        var = self.decode(var) if not tag else var * self.sh_rate
        # 计算目标值
        term1 = 100 * ((var[:, :self.dim - 1] ** 2 - var[:, 1:]) ** 2)  # 形状为 (n, dim-1)
        term2 = (var[:, :self.dim - 1] - 1) ** 2  # 形状为 (n, dim-1)
        obj = np.sum(term1 + term2, axis=1)  # 形状为 (n,)

        return obj


class Ackley(Task):
    def __init__(self, lb=-100, ub=100, dim=50, M=None, opt=None, shuffle=None, sh_rate=1.0):
        super().__init__(lb, ub, dim, M, opt, shuffle, sh_rate)

    def fnc(self, var, tag=False):
        # 变量预处理
        if not tag:
            var = self.decode(var)
        # 计算目标值
        avg_sum1 = np.sum(var ** 2, axis=1) / self.dim  # 每行平方和的均值，形状 (n,)
        avg_sum2 = np.sum(np.cos(2 * np.pi * var), axis=1) / self.dim  # 每行余弦和的均值，形状 (n,)
        obj = 20 + np.e - 20 * np.exp(-0.2 * np.sqrt(avg_sum1)) - np.exp(avg_sum2)

        return obj + 500  # 形状为 (n,)


class Weierstrass(Task):
    def __init__(self, lb=-100, ub=100, dim=50, M=None, opt=None, shuffle=None, sh_rate=0.5 / 100):
        super().__init__(lb, ub, dim, M, opt, shuffle, sh_rate)

    def fnc(self, var, tag=False):
        # 变量预处理
        var = self.decode(var) if not tag else var * self.sh_rate
        a, b, kmax = 0.5, 3, 20
        k = np.arange(kmax + 1)
        # 批量计算 Weierstrass 函数值
        temp = np.sum(
            a ** k * np.cos(2 * np.pi * b ** k * (var[:, :, np.newaxis] + 0.5)),
            axis=2
        )  # 形状为 (n, dim)
        term1 = np.sum(temp, axis=1)  # 形状为 (n,)
        # 常数项（标量）乘以样本数量 -> 向量
        constant = self.dim * np.sum(a ** k * np.cos(2 * np.pi * b ** k * 0.5))
        obj = term1 - constant  # 形状为 (n,)

        return obj + 600  # 形状为 (n,)


class Griewank(Task):
    def __init__(self, lb=-100, ub=100, dim=50, M=None, opt=None, shuffle=None, sh_rate=600.0 / 100):
        super().__init__(lb, ub, dim, M, opt, shuffle, sh_rate)

    def fnc(self, var, tag=False):
        # 变量预处理
        var = self.decode(var) if not tag else var * self.sh_rate
        # 计算目标值
        sum1 = np.sum(var ** 2, axis=1) / 4000  # 每行平方和除以4000，形状 (n,)
        prod = np.prod(np.cos(var / np.sqrt(np.arange(1, self.dim + 1))), axis=1)  # 每行余弦积，形状 (n,)
        obj = 1 + sum1 - prod

        return obj + 700  # 形状为 (n,)


class Rastrigin(Task):
    def __init__(self, lb=-100, ub=100, dim=50, M=None, opt=None, shuffle=None, sh_rate=5.12 / 100):
        super().__init__(lb, ub, dim, M, opt, shuffle, sh_rate)

    def fnc(self, var, tag=False):
        # 变量预处理
        var = self.decode(var) if not tag else var * self.sh_rate
        # 计算目标值
        obj = np.sum(var ** 2 - 10 * np.cos(2 * np.pi * var) + 10, axis=1)

        return obj  # 形状为 (n,)


class Schwefel(Task):
    def __init__(self, lb=-100, ub=100, dim=50, M=None, opt=None, shuffle=None, sh_rate=1000.0 / 100):
        super().__init__(lb, ub, dim, M, opt, shuffle, sh_rate)

    def fnc(self, var, tag=False):
        # 变量预处理
        var = var * self.sh_rate + 4.209687462275036e+002 if tag else self.decode(var) + 4.209687462275036e+002
        tmp = var.copy()  # 形状为 (n, dim)
        # 边界处理
        mask_low = var < -500
        var[mask_low] = -500 + np.fmod(np.abs(var[mask_low]), 500)
        mask_high = var > 500
        var[mask_high] = 500 - np.fmod(np.abs(var[mask_high]), 500)
        obj = (4.189828872724338e+002 * self.dim -
               np.sum(var * np.sin(np.sqrt(np.abs(var))), axis=1) +
               np.sum(((tmp + 500) ** 2) * mask_low, axis=1) / (10000 * self.dim) +
               np.sum(((tmp - 500) ** 2) * mask_high, axis=1) / (10000 * self.dim))

        return obj + 1100  # 形状为 (n,)


class Katsuura(Task):
    def __init__(self, lb=-100, ub=100, dim=50, M=None, opt=None, shuffle=None, sh_rate=5.0 / 100.0):
        super().__init__(lb, ub, dim, M, opt, shuffle, sh_rate)

    def fnc(self, var, tag=False):
        # 变量预处理
        var = var * self.sh_rate if tag else self.decode(var)
        tmp1 = np.power(2.0, np.arange(1, 33))  # 2^j，形状为 (32,)
        z_reshaped = var[:, :, np.newaxis]  # 形状为 (n, dim, 1)
        tmp2 = tmp1 * z_reshaped  # 形状为 (n, dim, 32)
        tmp3 = np.power(1.0 * self.dim, 1.2)
        temp = np.sum(np.abs(tmp2 - np.floor(tmp2 + 0.5)) / tmp1, axis=2)  # 形状为 (n, dim)
        weights = np.arange(1, self.dim + 1)  # 形状为 (dim,)
        f = np.prod(np.power(1.0 + weights * temp, 10.0 / tmp3, where=temp > 0), axis=1)  # 形状为 (n,)
        tmp1 = 10.0 / (self.dim * self.dim)
        obj = f * tmp1 - tmp1

        return obj  # 形状为 (n,)


class GrieRosen(Task):
    def __init__(self, lb=-100, ub=100, dim=50, M=None, opt=None, shuffle=None, sh_rate=5.0 / 100.0):
        super().__init__(lb, ub, dim, M, opt, shuffle, sh_rate)

    def fnc(self, var, tag=False):
        # 变量预处理
        var = var * self.sh_rate + 1 if tag else self.decode(var) + 1
        # 计算目标值
        temp = np.roll(var, -1, axis=1)  # 形状为 (n, dim)
        temp = 100 * (var ** 2 - temp) ** 2 + (var - 1) ** 2  # 形状为 (n, dim)
        obj = np.sum(temp ** 2 / 4000 - np.cos(temp) + 1, axis=1)  # 形状为 (n,)

        return obj + 1500  # 形状为 (n,)


class Escaffer6(Task):
    def __init__(self, lb=-100, ub=100, dim=50, M=None, opt=None, shuffle=None, sh_rate=1.0):
        super().__init__(lb, ub, dim, M, opt, shuffle, sh_rate)

    def fnc(self, var, tag=False):
        # 变量预处理
        if not tag:
            var = self.decode(var)
        # 计算目标值
        var_next = np.roll(var, -1, axis=1)
        temp1 = np.sin(np.sqrt(var ** 2 + var_next ** 2)) ** 2
        temp2 = 1.0 + 0.001 * (var ** 2 + var_next ** 2)
        terms = 0.5 + (temp1 - 0.5) / (temp2 ** 2)
        obj = np.sum(terms, axis=1)

        return obj + 1600  # 形状为 (n,)


class HappyCat(Task):
    def __init__(self, lb=-100, ub=100, dim=50, M=None, opt=None, shuffle=None, sh_rate=5.0 / 100.0):
        super().__init__(lb, ub, dim, M, opt, shuffle, sh_rate)

    def fnc(self, var, tag=False):
        # 变量预处理
        var = var * self.sh_rate - 1 if tag else self.decode(var) - 1
        # 计算目标值
        r2 = np.sum(var ** 2, axis=1)  # 每行平方和，形状为 (n,)
        sum_z = np.sum(var, axis=1)  # 每行和，形状为 (n,)
        obj = np.abs(r2 - self.dim) ** (1 / 4) + (0.5 * r2 + sum_z) / self.dim

        return obj + 1300 + 0.5  # 形状为 (n,)


class Hgbat(Task):
    def __init__(self, lb=-100, ub=100, dim=50, M=None, opt=None, shuffle=None, sh_rate=5.0 / 100.0):
        super().__init__(lb, ub, dim, M, opt, shuffle, sh_rate)

    def fnc(self, var, tag=False):
        # 变量预处理
        var = var * self.sh_rate - 1 if tag else self.decode(var) - 1
        # 计算目标值
        r2 = np.sum(var ** 2, axis=1)  # 每行平方和，形状为 (n,)
        sum_z = np.sum(var, axis=1)  # 每行和，形状为 (n,)
        obj = np.sqrt(np.abs(r2 ** 2 - sum_z ** 2)) + (r2 / 2 + sum_z) / self.dim

        return obj + 0.5  # 形状为 (n,)


class Hf01(Task):
    def __init__(self, lb=-100, ub=100, dim=50, M=None, opt=None, shuffle=None, sh_rate=1.0):
        super().__init__(lb, ub, dim, M, opt, shuffle, sh_rate)
        self.Sch = Schwefel(dim=15)
        self.Ras = Rastrigin(dim=15)
        self.Elp = Ellips(dim=20)

    def fnc(self, var):
        # 变量预处理
        var = self.decode(var)
        # 打乱维度索引
        temp = var[:, self.shuffle.astype(int) - 1]  # 形状为 (n, dim)
        # 计算子函数
        obj = (self.Sch.fnc(temp[:, :15], tag=True) +
               self.Ras.fnc(temp[:, 15:30], tag=True) +
               self.Elp.fnc(temp[:, 30:], tag=True))

        return obj + 600  # 形状为 (n,)


class Hf04(Task):
    def __init__(self, lb=-100, ub=100, dim=50, M=None, opt=None, shuffle=None, sh_rate=1.0):
        super().__init__(lb, ub, dim, M, opt, shuffle, sh_rate)
        self.Hg = Hgbat(dim=10)
        self.Dis = Discus(dim=10)
        self.GR = GrieRosen(dim=15)
        self.Ras = Rastrigin(dim=15)

    def fnc(self, var):
        # 变量预处理
        var = self.decode(var)
        # 打乱维度索引
        temp = var[:, self.shuffle.astype(int) - 1]  # 形状为 (n, dim)
        # 计算子函数
        obj = (self.Hg.fnc(temp[:, :10], tag=True) +
               self.Dis.fnc(temp[:, 10:20], tag=True) +
               self.GR.fnc(temp[:, 20:35], tag=True) +
               self.Ras.fnc(temp[:, 35:], tag=True))

        return obj + 500  # 形状为 (n,)


class Hf05(Task):
    def __init__(self, lb=-100, ub=100, dim=50, M=None, opt=None, shuffle=None, sh_rate=1.0):
        super().__init__(lb, ub, dim, M, opt, shuffle, sh_rate)
        self.Es = Escaffer6(dim=5)
        self.Hg = Hgbat(dim=10)
        self.Ros = Rosenbrock(dim=10)
        self.Sch = Schwefel(dim=10)
        self.Elp = Ellips(dim=15)

    def fnc(self, var):
        # 变量预处理
        var = self.decode(var)
        # 打乱维度索引
        temp = var[:, self.shuffle.astype(int) - 1]  # 形状为 (n, dim)
        # 计算子函数
        obj = (self.Es.fnc(temp[:, :5], tag=True) +
               self.Hg.fnc(temp[:, 5:15], tag=True) +
               self.Ros.fnc(temp[:, 15:25], tag=True) +
               self.Sch.fnc(temp[:, 25:35], tag=True) +
               self.Elp.fnc(temp[:, 35:], tag=True))

        return obj - 600  # 形状为 (n,)


class Hf06(Task):
    def __init__(self, lb=-100, ub=100, dim=50, M=None, opt=None, shuffle=None, sh_rate=1.0):
        super().__init__(lb, ub, dim, M, opt, shuffle, sh_rate)
        self.Kat = Katsuura(dim=5)
        self.HC = HappyCat(dim=10)
        self.GR = GrieRosen(dim=10)
        self.Sch = Schwefel(dim=10)
        self.Ack = Ackley(dim=15)

    def fnc(self, var):
        # 变量预处理
        var = self.decode(var)
        # 打乱维度索引
        temp = var[:, self.shuffle.astype(int) - 1]  # 形状为 (n, dim)
        # 计算子函数
        obj = (self.Kat.fnc(temp[:, :5], tag=True) +
               self.HC.fnc(temp[:, 5:15], tag=True) +
               self.GR.fnc(temp[:, 15:25], tag=True) +
               self.Sch.fnc(temp[:, 25:35], tag=True) +
               self.Ack.fnc(temp[:, 35:], tag=True))

        return obj - 2200  # 形状为 (n,)


path = os.path.abspath(os.path.dirname(__file__))  # 获取当前文件的绝对路径


# 读取文件
def GetMatrixs(filepath):
    filepath = path + filepath  # 构建文件的完整路径
    bias1 = np.loadtxt(filepath + 'bias_1')
    bias2 = np.loadtxt(filepath + 'bias_2')
    matrix1 = np.loadtxt(filepath + 'matrix_1')
    matrix2 = np.loadtxt(filepath + 'matrix_2')
    return {'bias1': bias1, 'bias2': bias2, 'ma1': matrix1, 'ma2': matrix2}


def GetShuffle(filepath):
    filepath = path + filepath  # 构建文件的完整路径
    shuffle = np.loadtxt(filepath)
    return {'shuffle': shuffle}


# 10个基本问题
def Benchmark1(filename='/Tasks/benchmark_1/'):
    params = GetMatrixs(filename)
    Problem.maxFE = 100 * 2000
    Problem.T = 2
    Task1 = Weierstrass(M=params['ma1'], opt=params['bias1'])
    Task2 = Weierstrass(M=params['ma2'], opt=params['bias2'])
    Probs = [Task1, Task2]
    return Probs


def Benchmark2(filename='/Tasks/benchmark_2/'):
    params = GetMatrixs(filename)
    Problem.maxFE = 100 * 2000
    Problem.T = 2
    Task1 = Griewank(M=params['ma1'], opt=params['bias1'])
    Task2 = Griewank(M=params['ma2'], opt=params['bias2'])
    Probs = [Task1, Task2]
    return Probs


def Benchmark3(filename='/Tasks/benchmark_3/'):
    params = GetMatrixs(filename)
    s = GetShuffle('/Tasks/shuffle/shuffle_data_17_D50.txt')
    Problem.maxFE = 100 * 2000
    Problem.T = 2
    Task1 = Hf01(M=params['ma1'], opt=params['bias1'], shuffle=s['shuffle'])
    Task2 = Hf01(M=params['ma2'], opt=params['bias2'], shuffle=s['shuffle'])
    Probs = [Task1, Task2]
    return Probs


def Benchmark4(filename='/Tasks/benchmark_4/'):
    params = GetMatrixs(filename)
    Problem.maxFE = 100 * 2000
    Problem.T = 2
    Task1 = HappyCat(M=params['ma1'], opt=params['bias1'])
    Task2 = HappyCat(M=params['ma2'], opt=params['bias2'])
    Probs = [Task1, Task2]
    return Probs


def Benchmark5(filename='/Tasks/benchmark_5/'):
    params = GetMatrixs(filename)
    Problem.maxFE = 100 * 2000
    Problem.T = 2
    Task1 = GrieRosen(M=params['ma1'], opt=params['bias1'])
    Task2 = GrieRosen(M=params['ma2'], opt=params['bias2'])
    Probs = [Task1, Task2]
    return Probs


def Benchmark6(filename='/Tasks/benchmark_6/'):
    params = GetMatrixs(filename)
    s = GetShuffle('/Tasks/shuffle/shuffle_data_21_D50.txt')
    Problem.maxFE = 100 * 2000
    Problem.T = 2
    Task1 = Hf05(M=params['ma1'], opt=params['bias1'], shuffle=s['shuffle'])
    Task2 = Hf05(M=params['ma2'], opt=params['bias2'], shuffle=s['shuffle'])
    Probs = [Task1, Task2]
    return Probs


def Benchmark7(filename='/Tasks/benchmark_7/'):
    params = GetMatrixs(filename)
    s = GetShuffle('/Tasks/shuffle/shuffle_data_22_D50.txt')
    Problem.maxFE = 100 * 2000
    Problem.T = 2
    Task1 = Hf06(M=params['ma1'], opt=params['bias1'], shuffle=s['shuffle'])
    Task2 = Hf06(M=params['ma2'], opt=params['bias2'], shuffle=s['shuffle'])
    Probs = [Task1, Task2]
    return Probs


def Benchmark8(filename='/Tasks/benchmark_8/'):
    params = GetMatrixs(filename)
    Problem.maxFE = 100 * 2000
    Problem.T = 2
    Task1 = Ackley(M=params['ma1'], opt=params['bias1'])
    Task2 = Ackley(M=params['ma2'], opt=params['bias2'])
    Probs = [Task1, Task2]
    return Probs


def Benchmark9(filename='/Tasks/benchmark_9/'):
    params = GetMatrixs(filename)
    Problem.maxFE = 100 * 2000
    Problem.T = 2
    Task1 = Schwefel(M=params['ma1'], opt=params['bias1'])
    Task2 = Escaffer6(M=params['ma2'], opt=params['bias2'])
    Probs = [Task1, Task2]
    return Probs


def Benchmark10(filename='/Tasks/benchmark_10/'):
    params = GetMatrixs(filename)
    Problem.maxFE = 100 * 2000
    Problem.T = 2
    s = GetShuffle('/Tasks/shuffle/shuffle_data_20_D50.txt')
    Task1 = Hf04(M=params['ma1'], opt=params['bias1'], shuffle=s['shuffle'])
    s = GetShuffle('/Tasks/shuffle/shuffle_data_21_D50.txt')
    Task2 = Hf05(M=params['ma2'], opt=params['bias2'], shuffle=s['shuffle'])
    Probs = [Task1, Task2]
    return Probs
