# <-*--*--*--*- Reference -*--*--*--*--*->
# @title: A Meta-Knowledge Transfer-Based Differential Evolution for Multitask Optimization
# @Author: Jian-Yu Li; Zhi-Hui Zhan; Kay Chen Tan; Jun Zhang
# @Journal: IEEE Transactions on Evolutionary Computation
# @year: 2022
# @Doi: 10.1109/TEVC.2021.3131236
# <-*--*--*--*- Coder -*--*--*--*--*->
# @Time: 2025/8/14 15:58
# @Author: wzb
# @Introduction: 多任务火星车导航问题的主函数

import os

from scipy.io import loadmat

from Problems.Problem import Problem
from Problems.RealWorld.Multiple_Robot_Navigation_Problem.rover_navigation import rover_navigation
from Problems.RealWorld.Multiple_Robot_Navigation_Problem.Point import Point
from Problems.RealWorld.Multiple_Robot_Navigation_Problem.Obstacle import Obstacle


def process_obstacles(arr):
    """
    处理障碍物数组，转换为 Obstacle 对象列表
    :param arr: 障碍物坐标数组
    :return: Obstacle 对象列表或 None
    """
    return [Obstacle(x, y) for x, y in zip(arr[:, 0], arr[:, 1])] if arr is not None else None


def process_point(arr):
    """
    处理点坐标，转换为 Point 对象
    :param arr: 点坐标数组
    :return: Point 对象或 None
    """
    return Point(arr[0, 0], arr[0, 1]) if arr is not None else None


def get_processor(name: str):
    """
    根据变量名前缀选择处理函数
    :param name: 变量名
    :return: 对应的处理函数
    """
    if name.startswith("obstacle"):
        return process_obstacles
    elif name.startswith("p_start") or name.startswith("p_goal"):
        return process_point
    else:
        return lambda arr: arr  # 默认：原样返回


def mat2python(filename, flags=None):
    """
    将 MATLAB .mat 文件转换为 Python 对象列表
    :param filename: .mat 文件名
    :param flags: 参数名列表（要提取的变量）；若为 None，则提取全部变量
    :return: 提取的参数字典 {变量名: 对象}
    """
    # 获取当前文件的绝对路径
    path = os.path.abspath(os.path.dirname(__file__))
    # 构建文件的完整路径
    file = path + filename
    data = loadmat(file)

    # 过滤掉 MATLAB 自带的字段（__header__, __version__, __globals__）
    variables = {k: v for k, v in data.items() if not k.startswith("__")}

    # 如果 flags=None，默认提取所有变量
    if flags is None:
        flags = variables.keys()

    # 遍历 flags 中的每个变量名，依次处理
    results = {}
    for name in flags:
        if name in variables:
            # 根据变量名获取对应的处理函数（如障碍物、起点、终点等）
            processor = get_processor(name)
            # 使用处理函数将原始数据转换为对应的 Python 对象
            results[name] = processor(variables[name])
        else:
            # 如果变量在 .mat 文件中不存在，则返回 None
            results[name] = None  # 如果变量不存在，则返回 None

    return results


def MRNP1(filename=r'/Maps/map_20_1.mat'):
    """
    障碍物20、维度25-25、地图1。
    """
    flags = ['obstacle1', 'p_start1', 'p_goal1', 'obstacle2', 'p_start2', 'p_goal2']
    params = mat2python(filename, flags)
    Problem.maxFE = 10 * 1000
    Problem.T = 2
    Task1 = rover_navigation(obstacle=params['obstacle1'], p_start=params['p_start1'], p_goal=params['p_goal1'], dim=25, lb=0, ub=1)
    Task2 = rover_navigation(obstacle=params['obstacle2'], p_start=params['p_start2'], p_goal=params['p_goal2'], dim=25, lb=0, ub=1)
    Probs = [Task1, Task2]
    return Probs


def MRNP2(filename=r'/Maps/map_20_2.mat'):
    """
    障碍物20、维度25-25、地图2。
    """
    flags = ['obstacle1', 'p_start1', 'p_goal1', 'obstacle2', 'p_start2', 'p_goal2']
    params = mat2python(filename, flags)
    Problem.maxFE = 10 * 1000
    Problem.T = 2
    Task1 = rover_navigation(obstacle=params['obstacle1'], p_start=params['p_start1'], p_goal=params['p_goal1'], dim=25, lb=0, ub=1)
    Task2 = rover_navigation(obstacle=params['obstacle2'], p_start=params['p_start2'], p_goal=params['p_goal2'], dim=25, lb=0, ub=1)
    Probs = [Task1, Task2]
    return Probs


def MRNP3(filename=r'/Maps/map_20_1.mat'):
    """
    障碍物20、维度50-50、地图1。
    """
    flags = ['obstacle1', 'p_start1', 'p_goal1', 'obstacle2', 'p_start2', 'p_goal2']
    params = mat2python(filename, flags)
    Problem.maxFE = 10 * 1000
    Problem.T = 2
    Task1 = rover_navigation(obstacle=params['obstacle1'], p_start=params['p_start1'], p_goal=params['p_goal1'], dim=50, lb=0, ub=1)
    Task2 = rover_navigation(obstacle=params['obstacle2'], p_start=params['p_start2'], p_goal=params['p_goal2'], dim=50, lb=0, ub=1)
    Probs = [Task1, Task2]
    return Probs


def MRNP4(filename=r'/Maps/map_20_2.mat'):
    """
    障碍物20、维度50-50、地图2。
    """
    flags = ['obstacle1', 'p_start1', 'p_goal1', 'obstacle2', 'p_start2', 'p_goal2']
    params = mat2python(filename, flags)
    Problem.maxFE = 10 * 1000
    Problem.T = 2
    Task1 = rover_navigation(obstacle=params['obstacle1'], p_start=params['p_start1'], p_goal=params['p_goal1'], dim=50, lb=0, ub=1)
    Task2 = rover_navigation(obstacle=params['obstacle2'], p_start=params['p_start2'], p_goal=params['p_goal2'], dim=50, lb=0, ub=1)
    Probs = [Task1, Task2]
    return Probs


def MRNP5(filename=r'/Maps/map_20_3.mat'):
    """
    障碍物20、维度50-50、地图3。
    """
    flags = ['obstacle1', 'p_start1', 'p_goal1', 'obstacle2', 'p_start2', 'p_goal2']
    params = mat2python(filename, flags)
    Problem.maxFE = 10 * 1000
    Problem.T = 2
    Task1 = rover_navigation(obstacle=params['obstacle1'], p_start=params['p_start1'], p_goal=params['p_goal1'], dim=50, lb=0, ub=1)
    Task2 = rover_navigation(obstacle=params['obstacle2'], p_start=params['p_start2'], p_goal=params['p_goal2'], dim=50, lb=0, ub=1)
    Probs = [Task1, Task2]
    return Probs


def MRNP6(filename=r'/Maps/map_20_4.mat'):
    """
    障碍物20、维度50-50、地图4。
    """
    flags = ['obstacle1', 'p_start1', 'p_goal1', 'obstacle2', 'p_start2', 'p_goal2']
    params = mat2python(filename, flags)
    Problem.maxFE = 10 * 1000
    Problem.T = 2
    Task1 = rover_navigation(obstacle=params['obstacle1'], p_start=params['p_start1'], p_goal=params['p_goal1'], dim=50, lb=0, ub=1)
    Task2 = rover_navigation(obstacle=params['obstacle2'], p_start=params['p_start2'], p_goal=params['p_goal2'], dim=50, lb=0, ub=1)
    Probs = [Task1, Task2]
    return Probs


def MRNP7(filename=r'/Maps/map_20_1.mat'):
    """
    障碍物20、维度100-100、地图1。
    """
    flags = ['obstacle1', 'p_start1', 'p_goal1', 'obstacle2', 'p_start2', 'p_goal2']
    params = mat2python(filename, flags)
    Problem.maxFE = 10 * 1000
    Problem.T = 2
    Task1 = rover_navigation(obstacle=params['obstacle1'], p_start=params['p_start1'], p_goal=params['p_goal1'], dim=100, lb=0, ub=1)
    Task2 = rover_navigation(obstacle=params['obstacle2'], p_start=params['p_start2'], p_goal=params['p_goal2'], dim=100, lb=0, ub=1)
    Probs = [Task1, Task2]
    return Probs


def MRNP8(filename=r'/Maps/map_20_2.mat'):
    """
    障碍物20、维度100-100、地图2。
    """
    flags = ['obstacle1', 'p_start1', 'p_goal1', 'obstacle2', 'p_start2', 'p_goal2']
    params = mat2python(filename, flags)
    Problem.maxFE = 10 * 1000
    Problem.T = 2
    Task1 = rover_navigation(obstacle=params['obstacle1'], p_start=params['p_start1'], p_goal=params['p_goal1'], dim=100, lb=0, ub=1)
    Task2 = rover_navigation(obstacle=params['obstacle2'], p_start=params['p_start2'], p_goal=params['p_goal2'], dim=100, lb=0, ub=1)
    Probs = [Task1, Task2]
    return Probs


def MRNP9(filename=r'/Maps/map_50_1.mat'):
    """
    障碍物50、维度50-50、地图1。
    """
    flags = ['obstacle1', 'p_start1', 'p_goal1', 'obstacle2', 'p_start2', 'p_goal2']
    params = mat2python(filename, flags)
    Problem.maxFE = 10 * 1000
    Problem.T = 2
    Task1 = rover_navigation(obstacle=params['obstacle1'], p_start=params['p_start1'], p_goal=params['p_goal1'], dim=50, lb=0, ub=1)
    Task2 = rover_navigation(obstacle=params['obstacle2'], p_start=params['p_start2'], p_goal=params['p_goal2'], dim=50, lb=0, ub=1)
    Probs = [Task1, Task2]
    return Probs


def MRNP10(filename=r'/Maps/map_50_2.mat'):
    """
    障碍物50、维度50-50、地图2。
    """
    flags = ['obstacle1', 'p_start1', 'p_goal1', 'obstacle2', 'p_start2', 'p_goal2']
    params = mat2python(filename, flags)
    Problem.maxFE = 10 * 1000
    Problem.T = 2
    Task1 = rover_navigation(obstacle=params['obstacle1'], p_start=params['p_start1'], p_goal=params['p_goal1'], dim=50, lb=0, ub=1)
    Task2 = rover_navigation(obstacle=params['obstacle2'], p_start=params['p_start2'], p_goal=params['p_goal2'], dim=50, lb=0, ub=1)
    Probs = [Task1, Task2]
    return Probs


def MRNP11(filename=r'/Maps/map_20_1.mat'):
    """
    障碍物20、维度50-25、地图1。
    """
    flags = ['obstacle1', 'p_start1', 'p_goal1', 'obstacle2', 'p_start2', 'p_goal2']
    params = mat2python(filename, flags)
    Problem.maxFE = 10 * 1000
    Problem.T = 2
    Task1 = rover_navigation(obstacle=params['obstacle1'], p_start=params['p_start1'], p_goal=params['p_goal1'], dim=50, lb=0, ub=1)
    Task2 = rover_navigation(obstacle=params['obstacle2'], p_start=params['p_start2'], p_goal=params['p_goal2'], dim=25, lb=0, ub=1)
    Probs = [Task1, Task2]
    return Probs


def MRNP12(filename=r'/Maps/map_20_2.mat'):
    """
    障碍物20、维度50-25、地图2。
    """
    flags = ['obstacle1', 'p_start1', 'p_goal1', 'obstacle2', 'p_start2', 'p_goal2']
    params = mat2python(filename, flags)
    Problem.maxFE = 10 * 1000
    Problem.T = 2
    Task1 = rover_navigation(obstacle=params['obstacle1'], p_start=params['p_start1'], p_goal=params['p_goal1'], dim=50, lb=0, ub=1)
    Task2 = rover_navigation(obstacle=params['obstacle2'], p_start=params['p_start2'], p_goal=params['p_goal2'], dim=25, lb=0, ub=1)
    Probs = [Task1, Task2]
    return Probs


def MRNP13(filename=r'/Maps/map_20_1.mat'):
    """
    障碍物20、维度25-50、地图1。
    """
    flags = ['obstacle1', 'p_start1', 'p_goal1', 'obstacle2', 'p_start2', 'p_goal2']
    params = mat2python(filename, flags)
    Problem.maxFE = 10 * 1000
    Problem.T = 2
    Task1 = rover_navigation(obstacle=params['obstacle1'], p_start=params['p_start1'], p_goal=params['p_goal1'], dim=25, lb=0, ub=1)
    Task2 = rover_navigation(obstacle=params['obstacle2'], p_start=params['p_start2'], p_goal=params['p_goal2'], dim=50, lb=0, ub=1)
    Probs = [Task1, Task2]
    return Probs


def MRNP14(filename=r'/Maps/map_20_2.mat'):
    """
    障碍物20、维度25-50、地图2。
    """
    flags = ['obstacle1', 'p_start1', 'p_goal1', 'obstacle2', 'p_start2', 'p_goal2']
    params = mat2python(filename, flags)
    Problem.maxFE = 10 * 1000
    Problem.T = 2
    Task1 = rover_navigation(obstacle=params['obstacle1'], p_start=params['p_start1'], p_goal=params['p_goal1'], dim=25, lb=0, ub=1)
    Task2 = rover_navigation(obstacle=params['obstacle2'], p_start=params['p_start2'], p_goal=params['p_goal2'], dim=50, lb=0, ub=1)
    Probs = [Task1, Task2]
    return Probs
