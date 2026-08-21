# <-*--*--*--*- Coder -*--*--*--*--*->
# @Time: 2025/4/14 下午8:02
# @Author: wzb
# @Introduction: 使用Wilcoxon秩和检验比较不同算法性能，并输出结果到Excel文件（显著性分析）
# @Remind: 需注意文件命名格式

import numpy as np
import pandas as pd
import scipy.stats as ss
from openpyxl.styles import Font

from Algorithms.Analyse.BasicOperations import BasicOperations
from Algorithms.Run.output import set_project_root

# 自动切换到项目根目录EMTO
project_root = set_project_root("EMTO")


def wilcoxon_rank_sum(file_path='', output_path='', isStd=False, isMaTOP=False, isScientificCounting = True, digits = 2):
    """
    使用Wilcoxon秩和检验比较不同算法性能，并输出结果到Excel文件（显著性分析）
    :param file_path: 输入文件路径
    :param output_path: 输出文件路径
    :param isStd: 输出结果是否包含标准差
    :param isMaTOP: 是否为超多任务优化处理（MaTOP）
    :param isScientificCounting: 是否使用科学计数法输出结果，默认True，False为小数点输出
    :param digits: 保留小数点位数/科学计数法有效位数，默认2位
    :return: Excel文件, 显著性分析结果
    """
    # 读取Excel文件和所有工作表名
    excel_file = pd.ExcelFile(file_path)
    sheet_names = excel_file.sheet_names

    # 原始算法数据（参考算法）
    df1 = pd.read_excel(file_path, sheet_name=sheet_names[0])
    col_data1 = np.column_stack([df1[col].values for col in df1.columns])
    # 所有问题数量
    num_algos = col_data1.shape[1]

    # 对比算法数据（对比算法）
    test_sheet_names = sheet_names[1:]
    num_tasks = len(test_sheet_names)

    # 显著性分析结果矩阵
    result = np.zeros((num_tasks, num_algos, 3), dtype=int)
    # 结果字符串矩阵
    results = np.empty((num_tasks, num_algos), dtype=object)

    for task_idx, sheet_name in enumerate(test_sheet_names):
        # 读取当前对比算法的数据
        df2 = pd.read_excel(file_path, sheet_name=sheet_name)
        col_data2 = np.column_stack([df2[col].values for col in df2.columns])

        # 遍历每个问题
        for alg_idx in range(num_algos):
            # 取参考算法和对比算法的同一列，长度取最小值，便于对齐比较数据
            length = min(len(col_data1[:, alg_idx]), len(col_data2[:, alg_idx]))
            col1 = col_data1[:, alg_idx][:length]
            col2 = col_data2[:, alg_idx][:length]
            mean_val2 = np.mean(col2)
            std_val2 = np.std(col2)
            # 输出均值或均值±标准差
            if isScientificCounting:
                # 科学计数法输出
                mean_str = f"{mean_val2:.{digits}E} ± {std_val2:.{digits}E}" if isStd else f"{mean_val2:.{digits}E}"
            else:
                # 小数点输出
                mean_str = f"{mean_val2:.{digits}f} ± {std_val2:.{digits}f}" if isStd else f"{mean_val2:.{digits}f}"
            # Wilcoxon秩和检验（两侧检验）
            stat, p_two = ss.ranksums(col1, col2, alternative='two-sided')
            if p_two >= 0.05:
                # 无显著差异，记为≈
                result[task_idx, alg_idx, 1] = 1
                mean_str += '(≈)'
            else:
                # 单侧检验，判断显著优劣
                stat, p_greater = ss.ranksums(col1, col2, alternative='greater')
                if p_greater < 0.05:
                    # 参考算法显著优于对比算法，记为(-)
                    result[task_idx, alg_idx, 2] = 1
                    mean_str += '(−)'
                else:
                    # 参考算法显著劣于对比算法，记为(+)
                    result[task_idx, alg_idx, 0] = 1
                    mean_str += '(+)'

            # 保存结果字符串
            results[task_idx, alg_idx] = mean_str

    # 构造输出矩阵（文本）
    output = np.empty((num_algos + 2, num_tasks + 1), dtype=object)
    output[0, 0] = sheet_names[0]
    output[0, 1:] = test_sheet_names

    # 数据行
    for alg_idx in range(num_algos):
        # 输出均值或均值±标准差
        output[alg_idx + 1, 0] = (
            # 科学计数法输出
            f"{np.mean(col_data1[:, alg_idx]):.{digits}E} ± {np.std(col_data1[:, alg_idx]):.{digits}E}" if isScientificCounting
            # 小数点输出
            else f"{np.mean(col_data1[:, alg_idx]):.{digits}f} ± {np.std(col_data1[:, alg_idx]):.{digits}f}"
        ) if isStd else (
            # 科学计数法输出
            f"{np.mean(col_data1[:, alg_idx]):.{digits}E}" if isScientificCounting
            # 小数点输出
            else f"{np.mean(col_data1[:, alg_idx]):.{digits}f}"
        )
        # 填充每个任务下该算法的显著性分析结果
        for task_idx in range(num_tasks):
            output[alg_idx + 1, task_idx + 1] = results[task_idx, alg_idx]

    # 汇总行（+/≈/-）
    summary_row = ["Number of + / ≈ / −"]
    for task_idx in range(num_tasks):
        counts = np.sum(result[task_idx, :, :], axis=0)
        summary_row.append(" / ".join(map(str, counts)))
    output[-1, :] = summary_row

    # 写入Excel
    processor = BasicOperations()
    # 设置字体，便于复制至word
    processor.cell_font = Font(name='Times New Roman', size=8)
    processor.title_font = Font(name='Times New Roman', size=8, bold=True)

    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # 显著性分析数据写入
        df_out = pd.DataFrame(output)
        df_out.to_excel(writer, index=False, header=False, sheet_name='显著性分析')
        # 超多任务处理器（统计每个问题的最终结果）
        if isMaTOP:
            MaTOP = analyze_results(result, test_sheet_names, group_sizes=np.full(shape=num_algos // 50, fill_value=50),
                                    threshold=5)
            df_MaTOP = pd.DataFrame(MaTOP)
            df_MaTOP.to_excel(writer, index=False, header=False, sheet_name='超多任务结果')
        # 优化表格
        processor.format_excel_file(writer)

    # 打印至控制台
    print(df_out)


def process_array(data_array: np.ndarray, group_sizes: np.ndarray, threshold: int = 5) -> np.ndarray:
    """
    处理输入数组，按指定分组（每个任务的函数数量）计算每组列和并标注差异符号。

    :param data_array: np.ndarray，输入的二维数组，行数需与分组总和一致
    :param group_sizes: np.ndarray，每组的行数数组（每个任务的函数数量）
    :param threshold: int，可选，差异阈值，默认为5
    :return: np.ndarray，包含每组列和及差异符号的字符串数组
    :raises ValueError: 如果分组总和与数组行数不匹配
    """
    # 计算分组的累计索引
    group_indices = np.concatenate(([0], np.cumsum(group_sizes)))
    total_rows = group_indices[-1]
    array_rows = data_array.shape[0]

    # 验证分组总和是否匹配数组行数
    if total_rows > array_rows:
        raise ValueError("分组数据的总和超过总问题行数")
    if total_rows < array_rows:
        raise ValueError("分组数据的总和少于总问题行数")

    # 初始化结果数组
    group_results = np.empty(len(group_sizes), dtype=object)

    # 遍历每一个问题，计算列和并生成结果字符串
    for i in range(len(group_sizes)):
        group = data_array[group_indices[i]:group_indices[i + 1]]
        col_sums = np.sum(group, axis=0)
        result_str = ' / '.join(str(sum_val) for sum_val in col_sums)
        # 显著好与显著差的差异值
        diff = col_sums[0] - col_sums[2]
        # 根据差异值添加符号
        if diff > threshold:
            result_str += ' (+)'
        elif diff < -threshold:
            result_str += ' (−)'
        else:
            result_str += ' (≈)'

        group_results[i] = result_str

    return group_results


def count_symbols(group_results: np.ndarray) -> str:
    """
    统计每行结果中的符号数量。

    :param group_results: np.ndarray，包含符号的字符串数组
    :return: str，格式为 "正符号数/约等符号数/负符号数" 的字符串
    """
    result_str = ' '.join(group_results)
    plus_count = result_str.count('(+)')
    minus_count = result_str.count('(−)')
    approx_count = result_str.count('(≈)')
    return f"{plus_count} / {approx_count} / {minus_count}"


def analyze_results(result_matrix: np.ndarray, test_sheet_names, group_sizes: np.ndarray, threshold: int) -> np.ndarray:
    """
    分析多组数据的结果，生成包含分组结果和符号统计的矩阵。

    :param result_matrix: np.ndarray，输入的三维数组，包含多组数据
    :param test_sheet_names: 标题行（表示算法名称）
    :param group_sizes: np.ndarray，每组的行数数组
    :param threshold: int，可选，差异阈值，默认为5
    :return: np.ndarray，包含分组结果和符号统计的矩阵
    """
    # 初始化信息行
    info = np.full((1, len(test_sheet_names)), fill_value='W / T / L', dtype=object)
    # 获取对比算法数量
    num_groups = result_matrix.shape[0]
    # 对每组数据应用 process_array 函数
    processed_results = np.array([
        process_array(result_matrix[i], group_sizes, threshold) for i in range(num_groups)
    ])
    # 计算每组的符号统计
    symbol_counts = np.array([count_symbols(row) for row in processed_results])
    # 合并标题行、信息行、结果和统计数据
    final_matrix = np.vstack([test_sheet_names, info, processed_results.T, symbol_counts])

    return final_matrix
