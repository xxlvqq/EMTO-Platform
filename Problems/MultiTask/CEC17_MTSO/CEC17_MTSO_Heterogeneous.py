# -*- coding: utf-8 -*-
"""
CEC17 异构维度多任务优化问题集（维度对齐对照实验专用）
=========================================================

背景：
    原 CEC17 MTSO 基准中两个任务维度相同（D=50），测试集里“异构维度”场景很少，
    难以体现 ATCTMTO 三种维度对齐策略（RDA 随机 / FDA 固定 / ZPA 零填充）的差异。
    本文件用 CEC17 单目标函数（Problems/Base.py）构造三组维度异构问题：

        轻度异构 Light  : T1 = 30,  T2 = 20
        中度异构 Medium : T1 = 50,  T2 = 25
        高度异构 High   : T1 = 100, T2 = 30

    每组内提供 2 个函数实例（Sphere 可分 / Rosenbrock 非可分），
    共 3 组 × 2 函数 = 6 个异构问题。

关键设计：
    1. 旋转矩阵 M = 1（单位阵），不做旋转。两个任务函数相同、维度不同，
       “前 min(d1,d2) 维一一对应”是自然的对齐方式（FDA 的假设）。
    2. 两个任务的全局最优 opt 取“完全交集”：opt2 = opt1 的前 dim2 维，即 Task2
       的最优解是 Task1 最优解在低维子空间上的投影，任务间存在真实可迁移的知识。
    3. opt 必须是“逐维不同”的非平凡向量。若 opt=0（各维最优相同），任一维度携带
       的“最优位置”信息完全相同，RDA/FDA/ZPA 将无法体现差异——这是本构造的核心。
    4. opt 向量由固定随机种子生成，跨变体 / 重复保持一致，保证公平对照。

注意事项（对齐策略何时真正起作用）：
    ATCTMTO 采用质心对齐 x' = x - Cs + Ct。对于可分函数（Sphere/Rastrigin/Elliptic），
    质心对齐会自动“校正”维度置换——无论源维度如何映射，结果都近似落在目标质心 Ct
    附近，因此三种策略在“缩减方向”（大维 → 小维）上差异很小；真正的差异集中在
    “扩展方向”（小维 → 大维）：RDA 对扩展维同样做质心校正，而 FDA/ZPA 用固定值
    0.5/0 填充扩展维。若要考察缩减方向上“固定映射 vs 随机映射”的差异，应使用
    非可分函数（Rosenbrock，或加入旋转矩阵）。

用法：
    from Problems.MultiTask.CEC17_MTSO.CEC17_MTSO_Heterogeneous import \
        HETEROGENEOUS_PROBLEMS, build_heterogeneous

    Prob = HETEROGENEOUS_PROBLEMS['Hetero_Light_Sphere']()
    Prob = build_heterogeneous(Sphere, 30, 20, -100, 100, seed=2025)
"""

import numpy as np

from Problems.Base import Rosenbrock, Sphere
from Problems.Problem import Problem


# ----------------------------------------------------------------------
#  三组异构维度配置（T1 维数, T2 维数）
# ----------------------------------------------------------------------
HETERO_LEVELS = {
    'Light':  (30, 20),   # 轻度异构
    'Medium': (50, 25),   # 中度异构
    'High':   (100, 30),  # 高度异构
}

# ----------------------------------------------------------------------
#  每组使用的 CEC17 函数：(类, 下界, 上界)
#  边界与 Problems/Base.py 中各函数的默认边界一致。
#  挑选 2 个代表性函数：Sphere（单峰可分）、Rosenbrock（非可分），
#  分别覆盖“质心对齐自动校正置换”与“置换破坏耦合”两种情形。
#  如需扩充，可在此字典追加 Problems.Base 中的其他函数，如 Rastrigin/Elliptic。
# ----------------------------------------------------------------------
CEC17_FUNCS = {
    'Sphere':     (Sphere,     -100.0, 100.0),  # 单峰可分
    'Rosenbrock': (Rosenbrock, -50.0,  50.0),   # 非可分
}


def build_heterogeneous(func_cls, dim1, dim2, lb, ub, seed=2025):
    """构造一个异构维度双任务问题（完全交集）。

    Task1 维度为 dim1，Task2 维度为 dim2（要求 dim2 <= dim1）。
    opt2 = opt1[:dim2]，保证两个任务在共享子空间上具有相同全局最优，
    从而存在真实的、按“前 k 维对应”的可迁移知识。

    :param func_cls: CEC17 单目标函数类（来自 Problems.Base）
    :param dim1: Task1 维度
    :param dim2: Task2 维度
    :param lb: 下界
    :param ub: 上界
    :param seed: opt 向量随机种子
    :return: [Task1, Task2]
    """
    rng = np.random.default_rng(seed)
    opt1 = rng.uniform(lb, ub, size=dim1)
    opt2 = opt1[:dim2].copy()

    Problem.maxFE = 100 * 1000
    Problem.T = 2
    Task1 = func_cls(M=1, opt=opt1, dim=dim1, lb=lb, ub=ub, g=0)
    Task2 = func_cls(M=1, opt=opt2, dim=dim2, lb=lb, ub=ub, g=0)
    return [Task1, Task2]


def _seed_for(level, func_name):
    """为 (异构程度, 函数) 生成确定性的 opt 随机种子。"""
    levels = list(HETERO_LEVELS.keys())
    funcs = list(CEC17_FUNCS.keys())
    return 1000 + levels.index(level) * 100 + funcs.index(func_name) + 1


def _make_factory(func_cls, dim1, dim2, lb, ub, seed):
    """返回一个无参工厂函数（便于注册表统一调用）。"""
    def factory():
        return build_heterogeneous(func_cls, dim1, dim2, lb, ub, seed)
    return factory


# ----------------------------------------------------------------------
#  问题注册表：name -> 无参工厂函数
#  共 3 组 × 2 函数 = 6 个问题，命名如 Hetero_Light_Sphere。
# ----------------------------------------------------------------------
HETEROGENEOUS_PROBLEMS = {}
for _level, (_d1, _d2) in HETERO_LEVELS.items():
    for _fname, (_cls, _lb, _ub) in CEC17_FUNCS.items():
        _name = f'Hetero_{_level}_{_fname}'
        _seed = _seed_for(_level, _fname)
        HETEROGENEOUS_PROBLEMS[_name] = _make_factory(_cls, _d1, _d2, _lb, _ub, _seed)

# 便于 `from ... import Hetero_Light_Sphere` 之类用法
for _name, _factory in HETEROGENEOUS_PROBLEMS.items():
    globals()[_name] = _factory
