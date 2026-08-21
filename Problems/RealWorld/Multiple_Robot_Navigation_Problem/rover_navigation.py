# <-*--*--*--*- Reference -*--*--*--*--*->
# @title: A Meta-Knowledge Transfer-Based Differential Evolution for Multitask Optimization
# @Author: Jian-Yu Li; Zhi-Hui Zhan; Kay Chen Tan; Jun Zhang
# @Journal: IEEE Transactions on Evolutionary Computation
# @year: 2022
# @Doi: 10.1109/TEVC.2021.3131236

# <-*--*--*--*- Coder -*--*--*--*--*->
# @Time: 2025/8/14 16:01
# @Author: wzb
# @Introduction: 多任务火星车导航问题的代价计算类

import numpy as np

from Problems.Problem import Problem
from Problems.RealWorld.Multiple_Robot_Navigation_Problem.Point import Point


class rover_navigation(Problem):
    """
    该类实现了火星车导航问题的代价计算方法。

    :param obstacle: Obstacle 对象列表，每个元素表示一个矩形障碍物，包含四个角点。
    :param p_start: 起点，Point 对象，具有 .x, .y 属性。
    :param p_goal: 终点，Point 对象，具有 .x, .y 属性。
    """

    def __init__(self, obstacle, p_start, p_goal, dim, lb, ub):
        super().__init__(dim, lb, ub)
        self.obstacle = obstacle
        self.p_start = p_start
        self.p_goal = p_goal

    def fnc(self, var):
        """
        计算给定折线路径的代价，包括路径长度、障碍物碰撞惩罚、起终点连接代价及常数项。

        :param var: N×dim 矩阵，每行表示一个个体路径的纵坐标序列。
        :return: obj: N×1 向量，每个个体的代价值；Con: N×1 全零向量（无约束）。
        """
        # 获取输入矩阵 var 的维度：N 为个体数量，dim 为路径点数
        N, dim = var.shape
        # 获取障碍物数量
        on = len(self.obstacle)
        # 计算 x 坐标步长
        x_step = 1 / (dim - 1)
        # 初始化代价向量 obj，N×1 全零
        obj = np.zeros(N)

        # 预计算障碍物的外包矩形 [xmin, xmax, ymin, ymax]
        obs_bbox = np.zeros((on, 4))
        # 遍历每个障碍物
        for j in range(on):
            # 提取障碍物四个角点的 x 坐标
            xs = [self.obstacle[j].angle[i].x for i in range(4)]
            # 提取障碍物四个角点的 y 坐标
            ys = [self.obstacle[j].angle[i].y for i in range(4)]
            # 计算障碍物的边界框：最小 x、最大 x、最小 y、最大 y
            obs_bbox[j, :] = [min(xs), max(xs), min(ys), max(ys)]

        # 遍历每个个体
        for k in range(N):
            # 构造路径点列表，x 坐标按步长均匀分布，y 坐标来自输入 var
            p = [Point(x_step * i, var[k, i]) for i in range(dim)]

            # 初始化碰撞惩罚
            punishment = 0
            # 初始化路径长度
            c_traj = 0

            # 遍历路径的每一段
            for i in range(dim - 1):
                # 计算当前路径段的边界框：最小 x
                seg_xmin = min(p[i].x, p[i + 1].x)
                # 计算当前路径段的边界框：最大 x
                seg_xmax = max(p[i].x, p[i + 1].x)
                # 计算当前路径段的边界框：最小 y
                seg_ymin = min(p[i].y, p[i + 1].y)
                # 计算当前路径段的边界框：最大 y
                seg_ymax = max(p[i].y, p[i + 1].y)

                # 粗检测：检查哪些障碍物的边界框与路径段边界框相交
                cand_idx = (obs_bbox[:, 0] <= seg_xmax) & (obs_bbox[:, 1] >= seg_xmin) & \
                           (obs_bbox[:, 2] <= seg_ymax) & (obs_bbox[:, 3] >= seg_ymin)

                # 精检测：对可能相交的障碍物进行碰撞检查
                for j in np.where(cand_idx)[0]:
                    # 检查路径段是否与障碍物碰撞
                    if self.is_collided(p[i], p[i + 1], self.obstacle[j]):
                        # 若碰撞，增加惩罚计数
                        punishment += 1

                # 累加路径段长度
                c_traj += self.dist(p[i], p[i + 1])

            # 计算总代价：路径长度 + 碰撞惩罚 + 起终点连接代价 + 常数项
            obj[k] = c_traj + 20 * punishment + \
                     10 * (self.dist(self.p_start, p[0]) + self.dist(self.p_goal, p[-1])) + 5

        # 返回代价向量
        return obj

    def is_collided(self, v_start, v_end, rect):
        """
        判断一条线段是否与矩形发生碰撞（包括端点在矩形内的情况）。

        :param v_start: 线段起点，Point 对象，具有 .x, .y 属性。
        :param v_end: 线段终点，Point 对象，具有 .x, .y 属性。
        :param rect: 矩形，Obstacle 对象，包含 angle[0..3] 四个角点。
        :return: flag: 1 表示碰撞，0 表示无碰撞。
        """
        # 检查线段起点或终点是否在矩形内
        if self.is_contains_point(v_start, rect) or self.is_contains_point(v_end, rect):
            # 若任一端点在矩形内，返回碰撞标志 1
            return 1
        # 检查线段是否与矩形的两条对角线相交
        return self.is_segment_intersects(v_start, v_end, rect.angle[0], rect.angle[2]) or \
            self.is_segment_intersects(v_start, v_end, rect.angle[1], rect.angle[3])

    def is_segment_intersects(self, v_start1, v_end1, v_start2, v_end2):
        """
        判断两条线段是否相交（包括端点重合或共线重叠）。

        :param v_start1: 第一条线段起点，Point 对象，具有 .x, .y 属性。
        :param v_end1: 第一条线段终点，Point 对象，具有 .x, .y 属性。
        :param v_start2: 第二条线段起点，Point 对象，具有 .x, .y 属性。
        :param v_end2: 第二条线段终点，Point 对象，具有 .x, .y 属性。
        :return: flag: 1 表示相交，0 表示不相交。
        """
        # 计算 v_start1 相对于 v_start2->v_end2 的左右性（叉积）
        left_s = self.point_is_left(v_start1, v_start2, v_end2)
        # 计算 v_end1 相对于 v_start2->v_end2 的左右性
        left_e = self.point_is_left(v_end1, v_start2, v_end2)
        # 若两点在同一侧且不共线，不相交
        if left_s * left_e > 0:
            return 0

        # 计算 v_start2 相对于 v_start1->v_end1 的左右性
        left_s = self.point_is_left(v_start2, v_start1, v_end1)
        # 计算 v_end2 相对于 v_start1->v_end1 的左右性
        left_e = self.point_is_left(v_end2, v_start1, v_end1)
        # 若两点在同一侧且不共线，不相交
        if left_s * left_e > 0:
            return 0

        # 其余情况（包括端点重合或共线重叠）视为相交
        return 1

    @staticmethod
    def point_is_left(v, v_start, v_end):
        """
        判断点 v 相对于有向线段 v_start -> v_end 的左右位置。

        :param v: 测试点，Point 对象，具有 .x, .y 属性。
        :param v_start: 线段起点，Point 对象，具有 .x, .y 属性。
        :param v_end: 线段终点，Point 对象，具有 .x, .y 属性。
        :return: val: >0 表示 v 在左侧，<0 表示右侧，=0 表示共线。
        """
        # 使用叉积计算点的左右性
        return (v_start.x - v.x) * (v_end.y - v.y) - (v_end.x - v.x) * (v_start.y - v.y)

    @staticmethod
    def is_contains_point(v, rect):
        """
        判断点 v 是否在矩形 rect 内部（含边界）。

        :param v: 测试点，Point 对象，具有 .x, .y 属性。
        :param rect: 矩形，Obstacle 对象，包含 angle[0..3] 四个角点。
        :return: flag: 1 表示点在矩形内或边界上，0 表示在外部。
        """
        # 提取矩形四个角点的 x 坐标
        xs = [rect.angle[i].x for i in range(4)]
        # 提取矩形四个角点的 y 坐标
        ys = [rect.angle[i].y for i in range(4)]
        # 计算矩形边界：最小 x
        left = min(xs)
        # 计算矩形边界：最大 x
        right = max(xs)
        # 计算矩形边界：最小 y
        bottom = min(ys)
        # 计算矩形边界：最大 y
        top = max(ys)

        # 检查点是否在矩形边界内（含边界）
        return 1 if (left <= v.x <= right) and (bottom <= v.y <= top) else 0

    @staticmethod
    def dist(p1, p2):
        """
        计算两点之间的欧几里得距离。

        :param p1: 第一个点，Point 对象，具有 .x, .y 属性。
        :param p2: 第二个点，Point 对象，具有 .x, .y 属性。
        :return: d: 两点之间的距离。
        """
        # 计算两点间的欧几里得距离
        return np.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)
