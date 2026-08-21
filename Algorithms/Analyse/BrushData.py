# <-*--*--*--*- Coder -*--*--*--*--*->
# @Time: 2025/7/23 12:00
# @Author: wzb
# @Introduction: 分析刷数据的工具，用于提取最优或最差的连续运行段
# @Remind: 1.请确保输入的 Excel 文件路径正确，且文件格式符合要求
#          2.处理后的文件格式统一为 EMTO 格式
#          3.该脚本可能会直接覆盖原有的 Excel 文件，请提前备份重要数据
import os

import numpy as np
import pandas as pd

from Algorithms.Analyse.BasicOperations import BasicOperations
from Algorithms.Run.output import set_project_root

# 项目根目录(项目根目录名称)
project_root = 'EMTO'
# 'EMTO': 行代表采样点，列代表运行次数，最后一列新增为平均值
# 'MTO-Platform'，行代表运行次数，列代表采样点
# 需要根据保存的文件格式进行选择，默认使用'EMTO'
save_the_format = 'EMTO'
# 是否更新保存的Excel文件
isUpdateExcel = True
# 取连续最好(True)还是最坏(False)
isBest = False
# 提取的连续最优运行段的长度
continuous_algebra = 30
# 优化的任务索引（从0开始）
task_idx = 0
# 待处理的文件路径
input_path = 'Files/MultiTask/MFEA/CEC/MTSO/CEC17_5/CI_HS.xlsx'
# 处理后的文件保存路径(默认(None)覆盖原文件，否则请指定新的文件路径)
output_path = None
output_path = input_path if output_path is None else output_path
os.makedirs(os.path.dirname(output_path), exist_ok=True)


def convert_T_to_numpy(T):
    """
    将输入的 DataFrame 列表转换为 numpy 矩阵列表，并删除每个矩阵的最后一列，适用于EMTO保留的文件格式。
    输入数据形状为 (采样数 + 1(表头), 运行次数 + 1(平均值))，输出数据形状为 (运行次数, 采样数)，即每一行每次运行的采样结果。

    :param T: 包含多个任务数据的 DataFrame 列表，每个 DataFrame 表示一个任务的采样结果
    :return: 转换后的 numpy 矩阵列表，每个矩阵的形状为 (运行次数, 采样数)
    """
    T_np = []
    for df in T:
        # 将 DataFrame 转换为 numpy 数组，并去掉最后一列
        arr = df.to_numpy()[:, :-1]
        # 转置矩阵，使其形状为 (运行次数, 采样数)
        T_np.append(arr.T)
    return T_np


def convert_to_numpy(T):
    """
    将输入的 DataFrame 列表转换为 numpy 矩阵列表，适用于MTO-Platform保留的文件格式。
    输入数据形状为 (运行次数, 采样数)，输出数据形状为 (运行次数, 采样数)，即每一行代表每次运行的采样结果。

    :param T: 包含多个任务数据的 DataFrame 列表，每个 DataFrame 表示一个任务的采样结果
    :return: 转换后的 numpy 矩阵列表，每个矩阵的形状为 (运行次数, 采样数)
    """
    T_np = []
    for df in T:
        # 将 DataFrame 转换为 numpy 数组，并去掉最后一列
        arr = df.to_numpy()
        # 转置矩阵，使其形状为 (运行次数, 采样数)
        T_np.append(arr)
    return T_np


def extract_continuous_best(T: list[np.ndarray], task_idx: int) -> list[np.ndarray]:
    """
    提取指定任务的最优连续运行段。

    :param T: 每个任务的采样结果数组列表
    :param task_idx: 优化的任务索引
    :return: 裁剪后的结果（每个任务保留连续 best runs）
    """
    # 获取指定任务的运行次数
    repeat = T[task_idx].shape[0]

    # 如果连续段长度（continuous_algebra）大于或等于运行次数，则直接返回原始数据
    if continuous_algebra >= repeat:
        print("Warning: continuous_algebra >= repeat, returning original data.")
        return T

    # 提取指定任务的最后一列数据（即每次运行的最终结果）
    final_results = T[task_idx][:, -1]
    # 使用滑动窗口计算连续段的和，窗口大小为 continuous_algebra
    window_sums = np.convolve(final_results, np.ones(continuous_algebra), mode='valid')
    # 根据 isBest 标志选择最优或最差的连续段起始索引
    if isBest:
        best_start_idx = np.argmin(window_sums)  # 最优：滑动窗口和的最小值
    else:
        best_start_idx = np.argmax(window_sums)  # 最差：滑动窗口和的最大值
    # 打印最佳连续段的起始索引及其平均值
    print(
        f"Task {task_idx + 1}: Best continuous segment starts at index {best_start_idx} with average {window_sums[best_start_idx] / continuous_algebra}")

    # 返回每个任务裁剪后的数据，仅保留最佳连续段
    return [t[best_start_idx:best_start_idx + continuous_algebra] for t in T]


def compute_average_convergence(T: list[np.ndarray]) -> list[np.ndarray]:
    """
    计算每个任务的平均收敛曲线。

    :param T: 每个任务的采样结果数组列表
    :return: 添加平均收敛后的新列表
    """
    for t in range(len(T)):
        # 计算当前任务在每个采样点上的平均值，得到平均收敛曲线
        mean_curve = np.mean(T[t], axis=0)
        # 将平均收敛曲线添加到当前任务的采样结果数组的最后一行
        T[t] = np.vstack([T[t], mean_curve])
    return T


def save_to_excel(T, name, num_tasks):
    """
    将结果保存为格式化的Excel文件。

    :param T: 每个任务的采样结果数组列表
    :param name: 问题名称
    :param num_tasks: 任务数量
    """
    # 检查是否需要更新保存的 Excel 文件（直接替换原来的数据文件）
    if not isUpdateExcel:
        print(f"{name} update failed!")
        return

    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        for t in range(num_tasks):
            # 将当前任务的 numpy 数组转置后转换为 DataFrame
            df = pd.DataFrame(T[t].T)
            df.to_excel(writer, sheet_name=f'T{t + 1}', index=False)
        # 对 Excel 文件进行格式化处理
        processor = BasicOperations()
        processor.format_excel_file(writer)

    print(f"{name} update completed! File has been saved to: {output_path}")  # 打印更新完成的信息，包含文件保存路径


def main():
    # 设置项目根目录
    set_project_root(project_root)
    # 提取问题名称（如：CI_HS）
    Name = os.path.splitext(os.path.basename(input_path))[0]
    # 提取任务数量
    t = len(pd.ExcelFile(input_path).sheet_names)
    # 提取数据
    T = list(pd.read_excel(input_path, sheet_name=None).values())
    # 将 Excel 文件中的数据转换为 numpy 数组
    T = convert_T_to_numpy(T) if save_the_format == 'EMTO' else convert_to_numpy(T)
    # 提取指定任务的最优连续运行段
    T = extract_continuous_best(T, task_idx)
    # 计算每个任务的平均收敛曲线，并将其添加到结果中
    T = compute_average_convergence(T)
    # 将处理后的数据保存为格式化的 Excel 文件
    save_to_excel(T, Name, t)


if __name__ == '__main__':
    main()
