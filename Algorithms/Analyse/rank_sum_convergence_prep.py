# <-*--*--*--*- Coder -*--*--*--*--*->
# @Time: 2025/4/14 下午8:02
# @Author: wzb
# @Introduction: 显著性分析和收敛图绘制的数据预处理
# @Remind: 运行前请确保路径准确且对应算法存在相应数据，修改数据后需删除算法.xlsx文件以重新生成数据。
# <-*--*--*--*- Use -*--*--*--*--*->
# 修改算法数据位置 folder_paths_template
# 修改文件保存路径 file
# 选择是否输出标准差 isStd 和 是否为超多任务优化处理 isMaTOP
# 运行

import os

import pandas as pd

from Algorithms.Analyse.BasicOperations import BasicOperations
from Algorithms.Analyse.convergence_plot import convergence_plot
from Algorithms.Analyse.wilcoxon_rank_sum import wilcoxon_rank_sum
from Algorithms.Run.output import set_project_root

# 配置变量
project_root = "EMTO"  # 项目根目录


def extract_algorithm_names(folder_paths):
    """
    从文件夹路径列表中提取算法名称，若有算法名称重复，则在名称后添加出现次数。

    :param folder_paths: list，文件夹路径列表
    :return: np.ndarray，包含算法名称的NumPy数组
    """
    result = []
    seen = {}  # 记录每个算法名称出现的次数

    for path in folder_paths:
        # 提取路径中的算法名称，假设算法名称在倒数第二个部分
        parts = path.split('/')
        algorithm = parts[-2]  # 倒数第二个部分通常是算法名称
        # 将下划线替换为连字符
        algorithm = algorithm.replace('_', '-')

        # 处理重复的算法名称
        if algorithm in seen:
            seen[algorithm] += 1
            result.append(f"{algorithm}-{seen[algorithm]}")
        else:
            seen[algorithm] = 0
            result.append(algorithm)

    return result


# 比较算法数据位置(该顺序决定了显著性分析和收敛图绘制的顺序，第一个为自己的算法)
folder_paths_template = [
    "Files/MultiTask/ATCTMTO/维度对齐/ATCTMTO/{testName}",
    "Files/MultiTask/ATCTMTO/维度对齐/fda/{testName}",
    "Files/MultiTask/ATCTMTO/维度对齐/zpa/{testName}",

    # "Files/MultiTask/ATCTMTO/组件分析/ATCTMTO/{testName}",
    # "Files/MultiTask/ATCTMTO/组件分析/ATCTMTO-NLPSR/{testName}",
    # "Files/MultiTask/ATCTMTO/组件分析/ATCTMTO-NCA/{testName}",
    # "Files/MultiTask/ATCTMTO/组件分析/ATCTMTO-FP/{testName}",

    # "Files/MultiTask/ATCTMTO/{testName}",
    # "Files/MultiTask/MFEA/{testName}",
    # "Files/MultiTask/EMEA/{testName}",
    # "Files/MultiTask/MFEA-AKT/{testName}",
    # "Files/MultiTask/AEMTO/{testName}",
    # "Files/MultiTask/RLMFEA/{testName}",
    # "Files/MultiTask/BLKT-DE/{testName}",
    # "Files/MultiTask/MTGA/{testName}",
    
    # 多种群算法对比
    # "Files/MultiTask/DDQN-RLMFEA/{testName}",
    # "Files/MultiTask/EMEA/{testName}",
    # "Files/MultiTask/MTGA/{testName}",
    # "Files/MTO/MultiTask/MTEA-AD/{testName}",
    # "Files/MultiTask/MKTDE/{testName}",
    # "Files/MTO/MultiTask/MTEA-SaO/{testName}",
    # "Files/MultiTask/AEMTO/{testName}",
    # "Files/MultiTask/BLKT-DE/{testName}",
    # "Files/MTO/MultiTask/MTEA-HKTS/{testName}",
    # "Files/MTO/MultiTask/SSLT-GA/{testName}",
    # # "Files/MTO/MultiTask/MTEA-PAE/{testName}",
    # "Files/MultiTask/PA-MTEA/{testName}",

    # 多因子算法对比
    # "Files/MultiTask/MFEA/{testName}",
    # "Files/MTO/MultiTask/MFEA-II/{testName}",
    # "Files/MultiTask/MFEA-AKT/{testName}",
    # "Files/MTO/MultiTask/ASCMFDE/{testName}",
    # "Files/MTO/MultiTask/MFEA-VC/{testName}",
    # "Files/MultiTask/RLMFEA/{testName}",
    # "Files/MultiTask/EMTO-AI/{testName}",
    # "Files/MultiTask/MFEA-ML/{testName}",

    # 组件分析
    # "Files/MultiTask/DDQN-RLMFEA/L2DQN-Comp/DDQN-MFEA/{testName}",
    # "Files/MultiTask/DDQN-RLMFEA/L2DQN-Comp/Comp-DE/{testName}",
    # "Files/MultiTask/DDQN-RLMFEA/L2DQN-Comp/Comp-GA/{testName}",
    # "Files/MultiTask/DDQN-RLMFEA/ktp-Comp/Comp-0.00/{testName}",
    # "Files/MultiTask/DDQN-RLMFEA/ktp-Comp/Comp-0.05/{testName}",
    # "Files/MultiTask/DDQN-RLMFEA/ktp-Comp/Comp-0.10/{testName}",
    # "Files/MultiTask/DDQN-RLMFEA/ktp-Comp/Comp-0.20/{testName}",
    # "Files/MultiTask/DDQN-RLMFEA/ktp-Comp/Comp-0.30/{testName}",
    # "Files/MultiTask/DDQN-RLMFEA/ktp-Comp/Comp-0.40/{testName}",
    # "Files/MultiTask/DDQN-RLMFEA/ktp-Comp/Comp-0.50/{testName}",
    # "Files/MultiTask/DDQN-RLMFEA/ktp-Comp/Comp-0.60/{testName}",
    # "Files/MultiTask/DDQN-RLMFEA/ktp-Comp/Comp-0.70/{testName}",
    # "Files/MultiTask/DDQN-RLMFEA/ktp-Comp/Comp-1.00/{testName}",



    # 比不过的算法
    # "Files/MTO/MultiTask/MTDE-ADKT/{testName}",
    # "Files/MTO/MultiTask/MTES-KG/{testName}",
    # "Files/MTO/MultiTask/MFMP/{testName}",
    # "Files/MultiTask/MMLMTO/{testName}",
    

    # 平台仓库down下来的
    # "Files/MTO/MultiTask/MTEA-AD/{testName}",
    # "Files/MTO/MultiTask/MTES/{testName}",
    # "Files/MultiTask/OTMTO/{testName}",
    # "Files/MultiTask/RLMFEA/{testName}",
    # "Files/MultiTask/BLKT-DE/{testName}",
    # "Files/MultiTask/EMTO-AI/{testName}",
    # "Files/MultiTask/MKTMTO/Comp/组件/STO/{testName}",

    # "Files/MultiTask/MFEA/{testName}",
    # "Files/MTO/MultiTask/MFEA/{testName}",
    # "Files/MTO/ManyTask/SBGA/{testName}",
    # "Files/MTO/ManyTask/MaTDE/{testName}",
    # "Files/MTO/MultiTask/MTEA-AD/{testName}",
    # "Files/MultiTask/AEMTO/{testName}",
    # "Files/MTO/ManyTask/BoKTGA/{testName}",
    # "Files/MTO/ManyTask/BoKTDE/{testName}",
    # "Files/MTO/ManyTask/TRADE/{testName}",
    # "Files/MTO/ManyTask/KR-MTEA/{testName}",
]
# 提取算法名称列表(如为[]则从文件夹路径模板自动提取(从文件夹路径中提取(倒数第二个 / 后的字符串)，并将下划线替换为连字符))
algos = []
if not algos:
    algos = extract_algorithm_names(folder_paths_template)

file = folder_paths_template[0]  # "Files/MultiTask/DDQN_RLMFEA/{testName}"  # 文件前缀
isStd = True  # 是否输出标准差
isMaTOP = False  # 是否为超多任务优化处理（MaTOP）
isScientificCounting = True  # 是否使用科学计数法输出结果
digits = 2  # 默认保留小数位数/ 科学计数法有效位数
data_file_template = "{file_prefix}/Data/{algo_name}_{data}_temp.xlsx"  # 显著性分析输出文件
plot_file_template = "{file_prefix}/Plot/{algo_name}_{data}_temp.xlsx"  # 收敛图输出文件


def process_excel_file(file_path, file_idx):
    """
    处理单个Excel文件，提取最后一列和最后一行数据

    :param file_path: str，Excel文件路径
    :param file_idx: int，文件索引
    :return:
        last_columns_data: list，包含最后一列数据的DataFrame列表
        last_rows_data: list，包含最后一行数据的DataFrame列表
    """
    last_columns_data = []
    last_rows_data = []

    if os.path.exists(file_path):
        excel_file = pd.ExcelFile(file_path)
        for sheet_idx, sheet_name in enumerate(excel_file.sheet_names):
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            col_name = f"P{file_idx + 1}-T{sheet_idx + 1}"
            # 提取最后一列，去掉NaN值
            last_column = df.iloc[:, -1].dropna()
            last_columns_data.append(pd.DataFrame({col_name: last_column}).reset_index(drop=True))
            # 提取最后一行（不含平均值列），去掉NaN值
            last_row = df.iloc[-1, :-1].dropna()
            last_rows_data.append(pd.DataFrame({col_name: last_row}).reset_index(drop=True))
    else:
        print(f"文件 {file_path} 不存在，{algos[file_idx]} 跳过处理。")

    return last_columns_data, last_rows_data


def check_sheet_exists(file_path, sheet_name):
    """
    检查指定Excel文件中是否存在指定工作表

    :param file_path: str，Excel文件路径
    :param sheet_name: str，工作表名称
    :return: bool，存在返回True，否则返回False
    """
    if not os.path.exists(file_path):
        print(f"文件 {file_path} 不存在，正在生成文件并处理数据...")
        return False
    xl = pd.ExcelFile(file_path)
    if sheet_name in xl.sheet_names:
        print(f"工作表 {sheet_name} 已存在于文件 {file_path} 中，直接利用已有数据。")
        return True
    else:
        print(f"工作表 {sheet_name} 不存在于文件 {file_path} 中，正在处理数据...")
        return False


def save_df_to_excel(dataframe, file_path, sheet_name, processor):
    """
    将DataFrame保存到Excel文件的指定工作表，并应用格式化处理。

    :param dataframe: pd.DataFrame，要保存的数据
    :param file_path: str，目标Excel文件路径
    :param sheet_name: str，目标工作表名称
    :param processor: BasicOperations，包含Excel格式化方法的处理器实例
    """
    # 如果目标文件已存在，则以追加模式写入（替换同名sheet）
    if os.path.exists(file_path):
        with pd.ExcelWriter(file_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
            dataframe.to_excel(writer, sheet_name=sheet_name, index=False)
            # 调用processor进行Excel格式化
            processor.format_excel_file(writer)
    else:
        # 文件不存在则新建文件并写入
        with pd.ExcelWriter(file_path, engine='openpyxl', mode='w') as writer:
            dataframe.to_excel(writer, sheet_name=sheet_name, index=False)
            # 调用processor进行Excel格式化
            processor.format_excel_file(writer)


def copy_sheet_simple(src_file, src_sheet, dst_file, dst_sheet, processor):
    """
    简单地将源Excel文件中的指定工作表复制到目标Excel文件的指定工作表，并进行格式化处理。

    :param src_file: str，源Excel文件路径
    :param src_sheet: str，源工作表名称
    :param dst_file: str，目标Excel文件路径
    :param dst_sheet: str，目标工作表名称
    :param processor: BasicOperations，包含Excel格式化方法的处理器实例
    """
    # 读取源sheet的数据为DataFrame
    df = pd.read_excel(src_file, sheet_name=src_sheet)
    # 保存DataFrame到目标文件的指定sheet，并格式化
    save_df_to_excel(dataframe=df, file_path=dst_file, sheet_name=dst_sheet, processor=processor)


def process_and_save_excel(folder_paths, file_order, data_file, plot_file, data, processor):
    """
    处理Excel数据并保存到输出文件，同时应用格式化

    :param folder_paths: list，文件夹路径列表
    :param file_order: list，文件处理顺序列表
    :param data_file: str，显著性分析输出文件路径
    :param plot_file: str，收敛图输出文件路径
    :param data: str，数据源名称
    :param processor: BasicOperations，基础操作处理器实例
    """
    # 遍历每个文件夹路径
    for folder_idx, folder_path in enumerate(folder_paths):
        # 数据文件的路径和工作表名称（Files/{AlgoType}/{AlgoName}/{testName}/{AlgoName}.xlsx）
        path = f"{folder_path}/{algos[folder_idx]}.xlsx"
        data_sheet_name = f"D_{data}"
        plot_sheet_name = f"P_{data}"

        # 检查工作表是否存在，若存在则直接使用已有数据，否则处理文件
        if not (check_sheet_exists(path, data_sheet_name) and check_sheet_exists(path, plot_sheet_name)):
            # 收敛图绘制数据
            last_columns_data = []
            # 显著性分析数据
            last_rows_data = []

            # 遍历文件获取每个文件的最后一列和最后一行数据
            for file_idx, filename in enumerate(file_order):
                full_filename = f"{filename}.xlsx"
                file_path = os.path.join(f"{folder_path}/{data}/", full_filename)
                col_data, row_data = process_excel_file(file_path, file_idx)
                last_columns_data.extend(col_data)
                last_rows_data.extend(row_data)

            row_datas = pd.concat(last_rows_data, axis=1)
            col_data = pd.concat(last_columns_data, axis=1)

            # 保存数据至Excel文件
            save_df_to_excel(dataframe=row_datas, file_path=path, sheet_name=data_sheet_name, processor=processor)
            save_df_to_excel(dataframe=col_data, file_path=path, sheet_name=plot_sheet_name, processor=processor)

        # 将处理后的数据复制到本次的显著性分析和收敛图的输出文件中
        copy_sheet_simple(src_file=path, src_sheet=data_sheet_name, dst_file=data_file, dst_sheet=algos[folder_idx],
                          processor=processor)
        copy_sheet_simple(src_file=path, src_sheet=plot_sheet_name, dst_file=plot_file, dst_sheet=algos[folder_idx],
                          processor=processor)

    print(f"处理完成！收敛图绘制数据保存至 {plot_file}，显著性分析数据保存至 {data_file}")


def main():
    """
    脚本入口函数
    """
    # 切换项目根目录
    set_project_root(project_root)
    # 初始化基础操作处理器，用于后续Excel格式化、数据选择、文件处理顺序获取等功能
    processor = BasicOperations()
    # 选择数据源
    testName, data = processor.select_data_source()
    # 获取文件处理顺序
    file_order = processor.get_processing_order(testName, data)

    # 根据所选数据更新路径和文件
    file_prefix = file.format(testName=testName)
    folder_paths = [path.format(data=data, testName=testName) for path in folder_paths_template]
    # 显著性分析和收敛图的输出文件路径
    data_file = data_file_template.format(data=data, file_prefix=file_prefix, algo_name=algos[0])
    plot_file = plot_file_template.format(data=data, file_prefix=file_prefix, algo_name=algos[0])

    # 创建必要的目录结构
    os.makedirs(f"{file_prefix}/Data", exist_ok=True)
    os.makedirs(f"{file_prefix}/Plot", exist_ok=True)
    # 删除以往的显著性和收敛图数据文件（如果存在）
    for temp_file in [data_file, plot_file]:
        if os.path.exists(temp_file):
            os.remove(temp_file)
        final_file = temp_file.replace("_temp", "")
        if os.path.exists(final_file):
            os.remove(final_file)

    # 处理数据并保存至对应路径的excel文件中
    process_and_save_excel(folder_paths, file_order, data_file, plot_file, data, processor)

    # 计算增量
    delta = int(data.split('_')[-1]) * 1000
    # 绘制收敛图和进行显著性分析
    convergence_plot(plot_file, plot_file.replace("_temp", ""), delta=delta)
    wilcoxon_rank_sum(file_path=data_file, output_path=data_file.replace("_temp", ""), isStd=isStd, isMaTOP=isMaTOP,
                      isScientificCounting=isScientificCounting, digits =digits)


if __name__ == "__main__":
    main()
