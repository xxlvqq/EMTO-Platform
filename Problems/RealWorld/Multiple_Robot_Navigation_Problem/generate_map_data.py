# <-*--*--*--*- Coder -*--*--*--*--*->
# @Time: 2025/8/14 18:35
# @Author: wzb
# @Introduction: 生成多任务火星车导航的2D地图数据，包括障碍物（中心点）、起点（横纵坐标）和终点（横纵坐标），保存为 .mat 文件。

import os

import numpy as np
import scipy


def generate_map_data(map_save_path, t_num, obstacle_num):
    """
    生成地图数据，包括障碍物（中心点）、起点（横纵坐标）和终点（横纵坐标），保存为 .mat 文件。

    参数：
    :param map_save_path: 保存 .mat 文件的路径。
    :param t_num: 任务数量。
    :param obstacle_num: 每个任务的障碍物数量。
    """
    data = {}

    for i in range(1, t_num + 1):
        # 生成障碍物
        obstacle_numeric = np.random.uniform(0.025, 0.975, size=(obstacle_num, 2))
        data[f'obstacle{i}'] = obstacle_numeric

        # 生成 p_start
        p_start = np.array([0, np.random.uniform(0, 1)])
        data[f'p_start{i}'] = p_start

        # 生成 p_goal
        p_goal = np.array([1, np.random.uniform(0, 1)])
        data[f'p_goal{i}'] = p_goal

    # 确保保存目录存在
    try:
        os.makedirs(os.path.dirname(map_save_path), exist_ok=True)
        scipy.io.savemat(map_save_path, data)
        print(f"地图数据已成功保存到：{map_save_path}")

    except Exception as e:
        print(f"地图数据保存失败，路径：{map_save_path}\n错误信息：{e}")


if __name__ == "__main__":
    T = 2  # 任务数量
    num_obstacle = 20  # 障碍物数量
    path = r'Maps/map_20_1.mat'
    abspath = os.path.abspath(os.path.dirname(__file__))  # 获取当前文件的绝对路径
    save_path = os.path.join(abspath, path)  # 构建文件的完整路径
    generate_map_data(save_path, T, num_obstacle)
