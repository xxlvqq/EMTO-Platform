# <-*--*--*--*- Coder -*--*--*--*--*->
# @Time: 2025/4/14 下午8:02
# @Author: wzb
# @Introduction: 基准函数
from abc import abstractmethod

import numpy as np

from Problems.Problem import Problem


class BaseFunction(Problem):
    """基准函数的基类，处理通用逻辑"""

    def __init__(self, M=1, opt=0, dim=50, lb=0, ub=1, g=0):
        super().__init__(dim, lb, ub)
        self.M = M
        self.opt = opt
        self.g = g

    def preprocess(self, var):
        """
        预处理变量：解码、平移、旋转
        :param var: 输入的变量矩阵，形状为 (n, dim)，每行是一个向量
        :return: 预处理后的变量矩阵，形状为 (n, dim)
        """
        # 处理标量 M 和 opt
        M = self.M * np.eye(self.dim) if np.isscalar(self.M) else self.M
        opt = self.opt * np.ones(self.dim) if np.isscalar(self.opt) else np.array(self.opt).flatten()
        # 确保 var 是二维数组
        var = np.atleast_2d(var)

        # 解码每行向量
        var = self.decode(var[:, :self.dim])
        # 平移：var - opt，opt 扩展为 (1, dim) 以广播到每行
        var = var - opt[np.newaxis, :]
        # 旋转：对每行应用矩阵 M，M 形状为 (dim, dim)
        var = np.array([np.dot(M, row) for row in var])

        return var

    @abstractmethod
    def fnc(self, var):
        """
        抽象方法，子类实现具体算法流程

        :param var: 种群基因型矩阵
        :return: Objs: 适应度值
        """
        raise NotImplementedError("Subclasses should implement this method")


class Ackley(BaseFunction):
    """Ackley 函数"""

    def __init__(self, M=1, opt=0, dim=50, lb=-50, ub=50, g=0):
        super().__init__(M, opt, dim, lb, ub, g)

    def fnc(self, var):
        var = self.preprocess(var)  # 形状为 (n, dim)
        # 计算每行的平方和与余弦和的均值
        avg_sum1 = np.mean(var ** 2, axis=1)  # 形状为 (n,)
        avg_sum2 = np.mean(np.cos(2 * np.pi * var), axis=1)  # 形状为 (n,)
        # 计算目标值
        obj = -20 * np.exp(-0.2 * np.sqrt(avg_sum1)) - np.exp(avg_sum2) + 20 + np.exp(1) + self.g
        return obj  # 形状为 (n,)


class Griewank(BaseFunction):
    """Griewank 函数"""

    def __init__(self, M=1, opt=0, dim=50, lb=-100, ub=100, g=0):
        super().__init__(M, opt, dim, lb, ub, g)

    def fnc(self, var):
        var = self.preprocess(var)  # 形状为 (n, dim)
        # 计算每行的平方和
        sum1 = np.sum(var ** 2, axis=1)  # 形状为 (n,)
        # 计算每行的余弦积
        sum2 = np.prod(np.cos(var / np.sqrt(np.arange(1, self.dim + 1))), axis=1)  # 形状为 (n,)
        # 计算目标值
        obj = 1 + sum1 / 4000 - sum2 + self.g
        return obj  # 形状为 (n,)


class Rastrigin(BaseFunction):
    """Rastrigin 函数"""

    def __init__(self, M=1, opt=0, dim=50, lb=-50, ub=50, g=0):
        super().__init__(M, opt, dim, lb, ub, g)

    def fnc(self, var):
        var = self.preprocess(var)  # 形状为 (n, dim)
        # 计算目标值
        obj = np.sum(var ** 2 - 10 * np.cos(2 * np.pi * var) + 10, axis=1) + self.g
        return obj  # 形状为 (n,)


class Rosenbrock(BaseFunction):
    """Rosenbrock 函数"""

    def __init__(self, M=1, opt=0, dim=50, lb=-50, ub=50, g=0):
        super().__init__(M, opt, dim, lb, ub, g)

    def fnc(self, var):
        var = self.preprocess(var)  # 形状为 (n, dim)
        # 计算相邻变量的差平方和
        term1 = 100 * (var[:, 1:] - var[:, :-1] ** 2) ** 2  # 形状为 (n, dim-1)
        term2 = (var[:, :-1] - 1) ** 2  # 形状为 (n, dim-1)
        obj = np.sum(term1 + term2, axis=1) + self.g
        return obj  # 形状为 (n,)


class Schwefel(BaseFunction):
    """Schwefel 函数"""

    def __init__(self, M=1, opt=0, dim=50, lb=-500, ub=500, g=0):
        super().__init__(M, opt, dim, lb, ub, g)

    def fnc(self, var):
        var = self.preprocess(var)  # 形状为 (n, dim)
        # 计算目标值
        obj = 418.9829 * self.dim - np.sum(var * np.sin(np.sqrt(np.abs(var))), axis=1) + self.g
        return obj  # 形状为 (n,)


class Schwefel2(BaseFunction):
    """Schwefel2 函数"""

    def __init__(self, M=1, opt=0, dim=50, lb=-50, ub=50, g=0):
        super().__init__(M, opt, dim, lb, ub, g)

    def fnc(self, var):
        var = self.preprocess(var)  # 形状为 (n, dim)
        # 计算每行的累积和的平方
        cumsum = np.cumsum(var, axis=1)  # 形状为 (n, dim)
        obj = np.sum(cumsum ** 2, axis=1) + self.g
        return obj  # 形状为 (n,)


class Sphere(BaseFunction):
    """Sphere 函数"""

    def __init__(self, M=1, opt=0, dim=50, lb=-100, ub=100, g=0):
        super().__init__(M, opt, dim, lb, ub, g)

    def fnc(self, var):
        var = self.preprocess(var)  # 形状为 (n, dim)
        # 计算每行的平方和
        obj = np.sum(var ** 2, axis=1) + self.g
        return obj  # 形状为 (n,)


class Weierstrass(BaseFunction):
    """Weierstrass 函数"""

    def __init__(self, M=1, opt=0, dim=50, lb=-0.5, ub=0.5, g=0):
        super().__init__(M, opt, dim, lb, ub, g)

    def fnc(self, var):
        var = self.preprocess(var)
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
        obj = term1 - constant + self.g

        return obj  # 形状为 (n,)


class Elliptic(BaseFunction):
    """Elliptic 函数"""

    def __init__(self, M=1, opt=0, dim=50, lb=-50, ub=50, g=0):
        super().__init__(M, opt, dim, lb, ub, g)

    def fnc(self, var):
        var = self.preprocess(var)  # 形状为 (n, dim)
        a = 1e6
        # 计算加权平方和
        if self.dim == 1:
            obj = var[:, 0] ** 2 + self.g  # 形状为 (n,)
        else:
            weights = a ** (np.arange(self.dim) / (self.dim - 1))  # 形状为 (dim,)
            obj = np.sum(weights * var ** 2, axis=1) + self.g
        return obj  # 形状为 (n,)
