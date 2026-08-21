# <-*--*--*--*- Coder -*--*--*--*--*->
# @Time: 2025/5/29 下午11:02
# @Author: wzb
# @Introduction: 将 MTO-Platform 的 MATLAB 数据转换为 Excel 格式
# @Remind: 正确设置路径和数据源，确保 MATLAB 文件存在且格式正确
#          mat文件只能包含一个测试集的数据(包含所有问题)，且文件按定义好的顺序命名
# <-*--*--*--*- Use -*--*--*--*--*->
# 修改数据源 data 和保存路径 file 中的算法名称后，直接运行即可
# MTO-Platform 平台代码运行指令
# mto({'XXX'},{'CEC17-MTSO1-CI-HS','CEC17-MTSO2-CI-MS','CEC17-MTSO3-CI-LS','CEC17-MTSO4-PI-HS','CEC17-MTSO5-PI-MS','CEC17-MTSO6-PI-LS','CEC17-MTSO7-NI-HS','CEC17-MTSO8-NI-MS','CEC17-MTSO9-NI-LS'},30,true,11,false,'XXX')
# mto({'XXX'},{'WCCI20-MTSO1','WCCI20-MTSO2','WCCI20-MTSO3','WCCI20-MTSO4','WCCI20-MTSO5','WCCI20-MTSO6','WCCI20-MTSO7','WCCI20-MTSO8','WCCI20-MTSO9','WCCI20-MTSO10'},30,true,11,false,'XXX')
# mto({'XXX'},{'WCCI20-MaTSO1','WCCI20-MaTSO2','WCCI20-MaTSO3','WCCI20-MaTSO4','WCCI20-MaTSO5','WCCI20-MaTSO6','WCCI20-MaTSO7','WCCI20-MaTSO8','WCCI20-MaTSO9','WCCI20-MaTSO10'},30,true,11,false,'XXX')
# mto({'XXX'},{'CEC19-MaTSO1','CEC19-MaTSO2','CEC19-MaTSO3','CEC19-MaTSO4','CEC19-MaTSO5','CEC19-MaTSO6'},30,true,11,false,'XXX')

import os
import numpy as np
import pandas as pd
import scipy.io as sio

from Algorithms.Analyse.BasicOperations import BasicOperations
from Algorithms.Run.output import set_project_root

project_root = "EMTO"  # 项目根目录

# 从指定路径加载 MATLAB 文件（.mat），文件包含多任务优化数据
data = sio.loadmat('Files/tmp/MTODataPKACK3.mat')

# 设置文件保存路径(基于类别和数据源，构建保存结果的目录结构)
# 路径格式：Files/ManyTask(测试集类型，如: MultiTask、ManyTask)/算法名称/测试集类别(CEC/Real、MTSO/MaTSO...)/测试集名称(CEC17_10、WCCI20_MTSO...)
file = "Files/MTO/MultiTask/SSLT-GA/{category}/{data_source}"
# 手动切换处理第几个算法(默认从0开始)的结果数据(多个算法结果数据存储在同一个 mat 文件中才需要设置)
num_algo = 0
# 自定义文件处理顺序，适用于只处理测试集部分任务，为[]时将使用处理器获取默认的处理顺序
file_names = []


def M2X(file_path, file_order, processor, num, num_finish):
    """
    将 MATLAB 的多任务优化结果数据提取并保存为 Excel 文件。

    :param file_path: str，保存结果的目录路径(包含类别和数据源)
    :param file_order: list，文件处理顺序列表（用于命名 Excel 文件）
    :param processor: BasicOperations 实例，用于 Excel 文件格式化等操作
    :param num: 当前需要处理的问题数量(从 MATLAB 数据中提取的结果数量)
    :param num_finish: 当前已处理的问题数量偏移量(用于从结果数据中正确索引问题)
    """
    # 提取 MATLAB 文件中的结果数据
    results = data['MTOData']['Results'][0, 0]
    # 获取结果数据的维度(num_problems: 问题数量; num_runs: 每次问题的运行次数)
    num_problems, _, num_runs = results.shape
    # 获取目标函数值的维度(从第一个问题的第一次运行中提取 'Obj' 数据，获取任务数（num_tasks）和保存结果数（num_objs）)
    num_tasks, num_objs = results[0, num_algo, 0]['Obj'].shape

    # 遍历每个问题
    for problem_idx in range(num):
        # 构建 Excel 文件保存路径(使用 file_order 中的文件名，确保文件名与处理顺序一致)
        excel_filename = f'{file_path}/{file_order[problem_idx]}.xlsx'

        # 为每个任务创建一个形状为 (num_objs, num_runs) 的零矩阵，用于存储目标值
        all_data = [np.zeros((num_objs, num_runs)) for _ in range(num_tasks)]

        # 遍历每次运行，收集所有运行的目标值数据
        for run in range(num_runs):
            # 提取当前问题、当前运行的目标值数据
            task_data = results[problem_idx + num_finish, num_algo, run]['Obj']
            # 遍历每个任务，将目标值转置后存入 all_data
            for task_idx in range(num_tasks):
                all_data[task_idx][:, run] = task_data[task_idx, :].T  # 转置以匹配维度

        # 创建 Excel 文件并写入数据
        with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
            # 遍历每个任务，计算均值并写入 Excel
            for task_idx in range(num_tasks):
                # 获取当前任务的所有运行数据
                current_data = all_data[task_idx]
                # 计算每行（目标）的均值，保持二维形状
                mean_cols = np.mean(current_data, axis=1, keepdims=True)
                # 将均值列追加到当前数据矩阵的末尾
                current_data = np.hstack((current_data, mean_cols))

                # 转换为 Pandas DataFrame
                # 列名为 0, 1, 2, ...，最后一列为均值
                columns = list(range(current_data.shape[1]))
                task_df = pd.DataFrame(current_data, columns=columns)

                # 写入 Excel 文件
                # 每个任务的数据写入单独的工作表，命名为 T1, T2, ...
                sheet_name = f'T{task_idx + 1}'
                task_df.to_excel(writer, sheet_name=sheet_name, index=False)

            # 调用处理器的方法，对 Excel 文件进行格式化（具体格式化逻辑由 processor 定义）
            processor.format_excel_file(writer)

    # 输出完成提示
    print(f'数据提取完成，所有 Excel 文件已生成，保存在：{os.path.abspath(file_path)}')


def main():
    """
    主函数入口
    """
    # 设置项目根目录
    set_project_root(project_root)
    # 初始化 BasicOperations 处理器，用于处理数据操作
    processor = BasicOperations()
    # 获取问题数量
    num_problems = data['MTOData']['Results'][0, 0].shape[0]
    # 获取问题名称和最大评估次数
    Probs = np.array([str(item[0]) for item in data['MTOData']['Problems'][0, 0]['Name'][0, :]])
    FEs = np.array([int(np.squeeze(item)) for item in data['MTOData']['Problems'][0, 0]['maxFE'][0, :]])

    # 已处理的任务数
    num_finish = 0

    # 如果没有问题，则直接返回
    while True:
        # 如果没有问题，则退出循环
        if num_problems == 0:
            print("没有待处理的问题，程序结束。")
            break
        print("---------------------")
        print("当前待处理的问题数量: ", num_problems)
        print("当前待处理的问题集: ", Probs[num_finish])
        print("当前待处理的问题评估次数: ", FEs[num_finish])
        # 调用处理器的方法，获取数据类别（如: CEC/MTSO...）和具体的数据源（如: CEC17_10）
        category, data_source = processor.select_data_source()
        # 根据数据类别和数据源，获取文件处理顺序（file_order），用于后续文件命名
        if not file_names:
            # 如果未指定 file_order，则使用处理器获取默认的处理顺序
            file_order = processor.get_processing_order(category, data_source)
        else:
            # 如果指定了 file_names，则使用指定的处理顺序
            file_order = file_names

        # 构建保存结果的目录路径
        file_path = file.format(category=category, data_source=data_source)
        # 创建保存结果的目录（如果不存在）
        os.makedirs(file_path, exist_ok=True)

        # 判断剩余问题数与文件顺序列表长度，确定本轮处理的问题数量
        if num_problems >= len(file_order):
            num = len(file_order)
        else:
            num = num_problems
        # 更新剩余待处理的问题数量
        num_problems -= num

        # 调用 M2X 函数，将 MATLAB 数据转换为 Excel 格式
        M2X(file_path, file_order, processor, num, num_finish)
        # 更新已处理的任务数
        num_finish += num


if __name__ == "__main__":
    main()
