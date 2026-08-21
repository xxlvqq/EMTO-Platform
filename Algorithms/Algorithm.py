# <-*--*--*--*- Coder -*--*--*--*--*->
# @Time: 2025/4/14 下午8:02
# @Author: wzb
# @Introduction: 算法基类，包含评估、终止检查等基本过程
# @Remind: 子类需实现 run() 方法；注意Problem类中 maxFE 和 T 的初始化

import numpy as np

from Problems.Problem import Problem


class Algorithm:
    def __init__(self):
        # 总评估次数
        self.FE = 0
        # 进化代数
        self.Gen = 0
        # 每一代的函数评估次数
        self.FE_Gen = []
        # 存储每个任务的最优适应度
        self.Best = np.full(Problem.T, np.inf)
        # 存储每个任务的最优个体位置
        self.BestInd = [np.array([]) for _ in range(Problem.T)]
        # 存储每一代每个任务的最优适应度
        self.Result = [[] for _ in range(Problem.T)]
        # 存储每一代每个任务的最优个体位置
        self.ResultInd = [[] for _ in range(Problem.T)]

    def reset(self):
        """
        重置算法状态
        """
        self.FE = 0
        self.Gen = 0
        self.FE_Gen = []
        self.Best = np.full(Problem.T, np.inf)
        self.BestInd = [np.array([]) for _ in range(Problem.T)]
        self.Result = [[] for _ in range(Problem.T)]
        self.ResultInd = [[] for _ in range(Problem.T)]

    def notTerminated(self, Prob, isPrint=False):
        """
        检查算法是否终止，并更新最佳个体的适应度和位置

        :param Prob: 问题对象
        :param isPrint: 是否打印中间过程，默认不打印
        :return: flag: 布尔值，True 表示未终止，False 表示已终止
        """
        # 每100代打印一次当前代数和每个任务的最优目标值
        if isPrint and len(self.FE_Gen) % 100 == 0 or self.FE >= Problem.maxFE:
            print('Generation ' + str(self.Gen) + ' :')
            print('Best objective of tasks : ', end='')
            print('[' + ' '.join([f'{val:.3E}' for val in self.Best]) + ']')

        if self.FE == 0:
            return True  # 初始状态未终止

        flag = self.FE < Problem.maxFE

        # 记录当前代的最优值与位置
        for t in range(Problem.T):
            self.Result[t].append(self.Best[t])
            self.ResultInd[t].append(self.BestInd[t])

        # 记录函数评估次数
        self.FE_Gen.append(self.FE)
        self.Gen += 1
        return flag

    def Evaluation(self, Pop, Prob, t):
        """
        评估种群的适应度，并更新最佳个体

        :param self: 算法对象
        :param Pop: 个体列表
        :param Prob: 问题对象
        :param t: 任务索引
        :return: flag: 任务改进标志（表示是否找到更优的个体）
        """
        # 初始化任务改进标志为 False
        flag = False
        # 更新函数评估次数
        self.FE += len(Pop)

        # 将种群中所有个体的编码向量抽取出来，组成一个 (n, dim) 的矩阵
        rnvecs = np.array([ind.rnvec for ind in Pop])  # shape: (个体数, 维度)
        # 使用问题对象计算种群中每个个体的适应度值
        Objs = Prob.fnc(rnvecs)  # shape: (n,)，每个个体的目标函数值
        # 遍历种群中的每个个体，更新其适应度值
        for idx, obj in enumerate(Objs):
            Pop[idx].obj = obj  # 更新个体的适应度

        # 获取当前种群中最优个体的索引（目标函数值最小）
        idx = np.argmin(Objs)
        # 如果当前最优个体优于历史记录，则更新最优记录并设置 flag 为 True
        if Pop[idx].obj < self.Best[t]:
            flag = True  # 表示此任务在本次进化中找到了更优个体
            self.Best[t] = Pop[idx].obj  # 更新该任务的最优适应度值
            self.BestInd[t] = Pop[idx].rnvec.copy()  # 保存该最优个体的基因型

        return flag

    def run(self, Prob):
        """
        抽象方法，子类实现具体算法流程

        :param Prob: 问题对象
        :return: None
        """
        raise NotImplementedError("Subclasses should implement this method")
