# <-*--*--*--*- Coder -*--*--*--*--*->
# @Time: 2025/8/14 16:00
# @Author: wzb
# @Introduction: 多任务火星车导航问题的障碍物类定义

from Problems.RealWorld.Multiple_Robot_Navigation_Problem.Point import Point


class Obstacle:

    def __init__(self, x, y, width=0.05, height=0.05):
        """
        初始化一个矩形障碍物对象。

        :param x: 障碍物中心点的 x 坐标。
        :param y: 障碍物中心点的 y 坐标。
        :param width: 障碍物的宽度，默认为 0.05。
        :param height: 障碍物的高度，默认为 0.05。
        """
        # 设置障碍物的中心点坐标和尺寸
        self.centre_x = x
        self.centre_y = y
        self.width = width
        self.height = height

        # 计算障碍物四个角点的坐标
        self.angle = [
            Point(x - self.width / 2, y + self.height / 2),  # Top-left corner
            Point(x + self.width / 2, y + self.height / 2),  # Top-right corner
            Point(x + self.width / 2, y - self.height / 2),  # Bottom-right corner
            Point(x - self.width / 2, y - self.height / 2)  # Bottom-left corner
        ]
