# <-*--*--*--*- Coder -*--*--*--*--*->
# @Time: 2025/4/14 下午8:02
# @Author: wzb
# @Introduction: 多因子个体基类，继承自Individual，用于管理多因子优化问题中的个体

import numpy as np
from Algorithms.Utils.Individual.Individual import Individual
from Problems.Problem import Problem


class Individual_MF(Individual):
    def __init__(self, MFCosts=None, MFRanks=None, MFFactor=None):
        """
        初始化多因子个体。

        :param MFCosts: 因子代价数组，默认初始化为长度为Problem.T的无穷大数组
        :param MFRanks: 因子排名数组，默认初始化为长度为Problem.T的None数组
        :param MFFactor: 技能因子，默认值为None
        """
        super().__init__()
        self.MFCosts = MFCosts if MFCosts is not None else np.full(Problem.T, np.inf)
        self.MFRanks = MFRanks if MFRanks is not None else np.array([None] * Problem.T)
        self.MFFactor = MFFactor
