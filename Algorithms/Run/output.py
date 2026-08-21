# <-*--*--*--*- Coder -*--*--*--*--*->
# @Time: 2025/4/14 下午8:02
# @Author: wzb
# @Introduction: 并行运行多任务优化算法，保留运行结果至相应的Excel文件
# @Remind: 运行前请确保参数均调整完毕
import argparse
import sys
import os as osi
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from openpyxl.styles import Font, Border, Side, Alignment


def set_project_root(target_folder_name: str = "EMTO") -> Path:
    """
    向上查找目标目录并切换为工作目录，同时返回该路径。
    可用于统一将当前目录切换至项目根目录。

    :param target_folder_name: 目标目录名称，默认为 "EMTO"
    :return: 查找到的目标目录路径（Path对象）
    :raises FileNotFoundError: 如果未找到目标目录
    """
    current_path = Path(__file__).resolve()
    for parent in current_path.parents:
        if parent.name == target_folder_name:
            sys.path.append(str(parent))
            osi.chdir(parent)
            return parent
    raise FileNotFoundError(f"项目根目录 '{target_folder_name}' 未找到")


# 自动切换到项目根目录EMTO
project_root = set_project_root("EMTO")

# 算法导入
from Algorithms.MultiTask.MultiFactorial.MFEA.MFEA import MFEA  # MFEA 算法
from Algorithms.MultiTask.MultiFactorial.MFDE.MFDE import MFDE  # MFDE 算法
from Algorithms.MultiTask.MultiFactorial.MFEA_AKT.MFEA_AKT import MFEA_AKT  # MFEA_AKT 算法
from Algorithms.MultiTask.MultiPopulation.EMEA.EMEA import EMEA  # EMEA 算法
from Algorithms.MultiTask.MultiFactorial.RLMFEA.RLMFEA import RLMFEA  # RLMFEA 算法
from Algorithms.MultiTask.MultiFactorial.EMTO_AI.EMTO_AI import EMTO_AI  # EMTO-AI 算法
from Algorithms.MultiTask.MultiPopulation.MTGA.MTGA import MTGA  # MTGA 算法
from Algorithms.MultiTask.MultiPopulation.MKTDE.MKTDE import MKTDE  # MKTDE 算法
from Algorithms.MultiTask.MultiPopulation.AEMTO.AEMTO import AEMTO  # AEMTO 算法
from Algorithms.MultiTask.MultiPopulation.BLKT_DE.BLKT_DE import BLKT_DE  # BLKT-DE 算法
from Algorithms.MultiTask.MultiPopulation.MMLMTO.MMLMTO import MMLMTO  # MMLMTO 算法
from Algorithms.SingleTask.Differential_Evolution.SHADE.SHADET import SHADET # SHADE 算法
from Algorithms.MultiTask.MultiFactorial.MFEA_ML.MFEA_ML import MFEA_ML  # MFEA-ML 算法
from Algorithms.MultiTask.MultiPopulation.PA_MTEA.PA_MTEA import PA_MTEA  # PA-MTEA 算法

from Algorithms.ME.DRL.DDQN.DDQN_RLMFEA import DDQN_RLMFEA    # DDQN算法
from Algorithms.ME.TSR_MFEA.TSR_MFEA import TSR_MFEA  # TSR-MFEA 算法
from Algorithms.ME.ATCTMTO import ATCTMTO  # ATCTMTO 算法
# from Algorithms.ME.ADLMTO.ADLMTO import ADLMTO # ADLMTO 算法
# from Algorithms.MultiTask.MultiFactorial.STO.STO import STO # STO 算法
# from Algorithms.MultiTask.MultiPopulation.OTMTO.OTMTO import OTMTO # OTMTO 算法
# from Algorithms.MultiTask.MultiFactorial.STO.STDE import STDE # STDE 算法

# 基类导入
from Problems.Problem import Problem

# 测试集导入
from Problems.MultiTask.CEC17_MTSO.CEC17_MTSO import *
from Problems.MultiTask.WCCI20_MTSO.WCCI20_MTSO import *
from Problems.MultiTask.Competitive_C2TOP.CEC17_MTSO_Competitive import *
from Problems.ManyTask.CEC19_MaTSO.CEC19_MaTSO import CEC19_MaTSO
from Problems.ManyTask.WCCI20_MaTSO.WCCI20_MaTSO import WCCI20_MaTSO
from Problems.RealWorld.Parameter_Extraction_of_Photovoltaic_Models.PEPVM import PEPVM
from Problems.RealWorld.Planar_Kinematic_Arm_Control_Problem.PKACP import PKAC
from Problems.RealWorld.Multiple_Robot_Navigation_Problem.MRNP import *

# 算法列表
Algos = [MFEA, MFDE, MFEA_AKT, RLMFEA, EMTO_AI, MTGA, MKTDE, AEMTO, BLKT_DE, MMLMTO, EMEA, SHADET, MFEA_ML, 
         DDQN_RLMFEA, TSR_MFEA, ATCTMTO, PA_MTEA]  
        #, ADLMTO, STO, OTMTO, STDE]
strAlgos = ['MFEA', 'MFDE', 'MFEA-AKT', 'RLMFEA', 'EMTO-AI', 'MTGA', 'MKTDE', 'AEMTO', 'BLKT-DE', 'MMLMTO', 'EMEA',
            'SHADE','MFEA-ML', 'DDQN-RLMFEA','TSR_MFEA', 'ATCTMTO', 'PA-MTEA',
            'STO', 'OTMTO', 'STDE', 'ADLMTO']  # 算法名称列表

# 设置参数
labs = ['', 'STO', 'Comp', 'Param', 'ScaleTest', 'Real', 'Strategy',
        'Other']  # 实验名称列表([0: CEC测试集, 1: STO, 2: 组件分析, 3: 参数调研, 4: 扩展性研究, 5: 真实世界问题, 6: 策略研究, 7: 其他])
labs_idx = 7  # 选择的实验索引([0: CEC测试集, 1: STO, 2: 组件分析, 3: 参数调研, 4: 扩展性研究, 5: 真实世界问题, 6: 策略研究, 7: 其他])
sub_exp_name = ''  # 具体实验问题名称(CEC测试集、STO、真实世界问题可为空，将作为保存文件的文件夹名称，故须符合文件夹命名规范)，表示具体实验下的具体问题名称(将'/'作为起始)，如：/没有策略1(组件分析)、/参数1/参数1取值(参数调研)、/扩展维度(扩展性问题)、/具体策略(使用不同策略)等
algo_idx = 15  # 算法索引([0: MFEA, 1: MFDE, 2: MFEA_AKT, 3: RLMFEA, 4: EMTO_AI, 5: MTGA, 6: MKTDE, 7: AEMTO, 
              # 8: BLKT_DE, 9: MMLMTO, 10: EMEA, 11: SHADE, 12: MFEA-ML, 13: DDQN-RLMFEA, 14: TSR_MFEA, 15: ATCTMTO, 16: PA_MTEA
              
              # ###原版##13: STO, 14: OTMTO, 15: STDE])
the_type = 'MultiTask'  # 算法类型（'MultiTask'：多任务算法，'ManyTask'：超多任务算法）
maxFE = 100 * 10 * 100  # 最大函数评估次数(None: 表示使用默认值(CEC17: 10万, CEC22: 20万, C2TOP: 10万, CEC19_MaTSO: 500万, WCCI20_MaTSO: 500万, PEPVM: 30万, PKACP: T * N * Gen万, MRNP: 1万))
repeat = 30  # 重复运行次数
continuous_algebra = 30  # 连续代数，用于提取最优连续段
task_idx = 0  # 优化任务的索引
isUpdateExcel = True  # 是否更新保存到Excel文件
isPrint = False  # 是否打印运行过程
sample_points = 11  # 采样点数量
output_dir = f"Files/{the_type}/{strAlgos[algo_idx]}"  # 输出目录主路径(最终输出路径：Files/MultiTask(ManyTask)/算法名称/实验名称/具体实验问题名称(可空)/测试集名称_评估次数(×万,每个任务的评估次数).xlsx)


def select_problem(func_id: int, dim: int = None) -> Tuple:
    """
    根据功能编号选择问题及其名称。

    :param func_id: 功能编号(0-8: CEC17, 9-18: CEC22, 19-27: C2TOP, 28-33: CEC19_MaTSO, 34-43: WCCI20_MaTSO, 44: PEPVM, 45: PKACP, 46-59: MRNP)
    :param dim: 维度参数(仅对 PKACP 等支持维度设置的问题有效；为 None 时使用默认值 50)
    :return: 问题实例、问题名称
    运行示例1(除PKACP):  python ./Algorithms/Run/output.py --func 45
    运行示例2(PKACP):    python ./Algorithms/Run/output.py --func 45 --dim 500
    """
    if func_id < 9:
        Probs17 = [CI_HS(), CI_MS(), CI_LS(), PI_HS(), PI_MS(), PI_LS(), NI_HS(), NI_MS(), NI_LS()]
        str17 = ['CI_HS', 'CI_MS', 'CI_LS', 'PI_HS', 'PI_MS', 'PI_LS', 'NI_HS', 'NI_MS', 'NI_LS']
        return Probs17[func_id], str17[func_id], 'CEC/MTSO/CEC17'
    elif func_id < 19:
        Probs22 = [Benchmark1(), Benchmark2(), Benchmark3(), Benchmark4(), Benchmark5(), Benchmark6(), Benchmark7(),
                   Benchmark8(), Benchmark9(), Benchmark10()]
        str22 = ['Benchmark1', 'Benchmark2', 'Benchmark3', 'Benchmark4', 'Benchmark5', 'Benchmark6', 'Benchmark7',
                 'Benchmark8', 'Benchmark9', 'Benchmark10']
        return Probs22[func_id - 9], str22[func_id - 9], 'CEC/MTSO/CEC22'
    elif func_id < 28:
        case = 3  # 案例编号，用于选择C2TOP问题集
        ProbsC2TOP = [C_CI_HS(case), C_CI_MS(case), C_CI_LS(case), C_PI_HS(case), C_PI_MS(case), C_PI_LS(case),
                      C_NI_HS(case),
                      C_NI_MS(case), C_NI_LS(case)]
        strC2TOP = ['C_CI_HS', 'C_CI_MS', 'C_CI_LS', 'C_PI_HS', 'C_PI_MS', 'C_PI_LS', 'C_NI_HS', 'C_NI_MS', 'C_NI_LS']
        return ProbsC2TOP[func_id - 19], strC2TOP[func_id - 19], f"C2TOP_{case}"
    elif func_id < 34:
        Probs19MaTSO = [CEC19_MaTSO(1), CEC19_MaTSO(2), CEC19_MaTSO(3), CEC19_MaTSO(4), CEC19_MaTSO(5), CEC19_MaTSO(6)]
        str19MaTSO = ['CEC19_MaTSO1', 'CEC19_MaTSO2', 'CEC19_MaTSO3', 'CEC19_MaTSO4', 'CEC19_MaTSO5', 'CEC19_MaTSO6']
        return Probs19MaTSO[func_id - 28], str19MaTSO[func_id - 28], 'CEC/MaTSO/CEC19'
    elif func_id < 44:
        Probs20MaTSO = [WCCI20_MaTSO(1), WCCI20_MaTSO(2), WCCI20_MaTSO(3), WCCI20_MaTSO(4), WCCI20_MaTSO(5),
                        WCCI20_MaTSO(6), WCCI20_MaTSO(7), WCCI20_MaTSO(8), WCCI20_MaTSO(9), WCCI20_MaTSO(10)]
        str20MaTSO = ['WCCI20_MaTSO1', 'WCCI20_MaTSO2', 'WCCI20_MaTSO3', 'WCCI20_MaTSO4', 'WCCI20_MaTSO5',
                      'WCCI20_MaTSO6', 'WCCI20_MaTSO7', 'WCCI20_MaTSO8', 'WCCI20_MaTSO9', 'WCCI20_MaTSO10']
        return Probs20MaTSO[func_id - 34], str20MaTSO[func_id - 34], 'CEC/MaTSO/CEC20'
    elif func_id < 45:
        ProbsPEPVM = [PEPVM()]
        strPEPVM = ['PEPVM']
        return ProbsPEPVM[0], strPEPVM[0], 'PEPVM/PEPVM'
    elif func_id < 46:
        # PKACP: 支持通过 --dim 参数设置维度，默认维度可设置
        d = dim if dim is not None else 50
        ProbsPKACP = [PKAC(N=50, T=2, Gen=100, dim=d, lb=0.0, ub=1.0)]
        strPKACP = [f'd{d}'] if dim is not None else ['PKACP']
        return ProbsPKACP[0], strPKACP[0], 'PKACP/PKACP'
        # ProbsPKACP = [PKAC(N=50, T=2, Gen=100, dim=500, lb=0.0, ub=1.0)] 
        # strPKACP = ['PKACP']
        # return ProbsPKACP[0], strPKACP[0], 'PKACP/PKACP'
    elif func_id < 60:
        ProbsMRNP = [MRNP1(), MRNP2(), MRNP3(), MRNP4(), MRNP5(), MRNP6(), MRNP7(), MRNP8(), MRNP9(), MRNP10(),
                     MRNP11(), MRNP12(),
                     MRNP13(), MRNP14()]
        strMRNP = ['MRNP1', 'MRNP2', 'MRNP3', 'MRNP4', 'MRNP5', 'MRNP6', 'MRNP7', 'MRNP8', 'MRNP9', 'MRNP10', 'MRNP11',
                   'MRNP12',
                   'MRNP13', 'MRNP14']
        return ProbsMRNP[func_id - 46], strMRNP[func_id - 46], 'MRNP/MRNP'
    else:
        raise ValueError(f"无效的功能编号: {func_id}")


def run_algorithm(prob, name: str) -> List[np.ndarray]:
    """
    运行多任务优化算法并收集结果。

    :param prob: 多任务优化问题实例
    :param name: 问题名称，用于输出和日志
    :return: 每个任务的采样结果数组列表（每个元素 shape: (repeat, sample_points)）
    """
    Problem.maxFE = maxFE if maxFE is not None else Problem.maxFE
    T = [np.zeros((repeat, sample_points)) for _ in range(Problem.T)]
    print(f"{name} initialization completed!")
    for i in range(repeat):
        result = Algos[algo_idx]().run(prob, isPrint=isPrint)  # 重新初始化算法实例，并运行
        gen_best = result.Result

        for t in range(Problem.T):
            gen_best_array = np.array(gen_best[t])
            indices = np.linspace(0, len(gen_best_array) - 1, sample_points, dtype=int)
            T[t][i] = gen_best_array[indices]

        print(f"{name} run {i + 1} completed!")
        log_results(T, name, i + 1, Problem.T)

    print(f"{name} run completed!")
    return T


def log_results(T: List[np.ndarray], name: str, run_idx: int, num_tasks: int):
    """
    记录当前运行的每个任务最优值和平均值。

    :param T: 每个任务的采样结果数组列表
    :param name: 问题名称
    :param run_idx: 当前运行索引
    :param num_tasks: 任务数量
    """
    print(f"\n{name} Values of the previous {run_idx} generation:")
    for r in range(run_idx):
        best_values = [T[t][r, -1] for t in range(num_tasks)]
        print(" ".join(f"{value:.2E}" for value in best_values))

    print(f"{name} Average Values of the previous {run_idx} generation:")
    avg_values = [
        np.mean(T[t][:run_idx, -1]) for t in range(num_tasks)
    ]
    print(" ".join(f"{value:.2E}" for value in avg_values))
    print()


def extract_continuous_best(T: List[np.ndarray], task_idx: int) -> List[np.ndarray]:
    """
    提取指定任务的最优连续运行段。

    :param T: 每个任务的采样结果数组列表
    :param task_idx: 优化的任务索引
    :return: 裁剪后的结果（每个任务保留连续 best runs）
    """
    if continuous_algebra >= repeat:
        return T

    final_results = T[task_idx][:, -1]
    window_sums = np.convolve(final_results, np.ones(continuous_algebra), mode='valid')
    best_start_idx = np.argmin(window_sums)

    return [t[best_start_idx:best_start_idx + continuous_algebra] for t in T]


def compute_average_convergence(T: List[np.ndarray]) -> List[np.ndarray]:
    """
    计算每个任务的平均收敛曲线。

    :param T: 每个任务的采样结果数组列表
    :return: 添加平均收敛后的新列表
    """
    for t in range(len(T)):
        mean_curve = np.mean(T[t], axis=0)
        T[t] = np.vstack([T[t], mean_curve])
    return T


def save_to_excel(T: List[np.ndarray], name: str, num_tasks: int, testName: str) -> None:
    """
    将结果保存为格式化的Excel文件。

    :param T: 每个任务的采样结果数组列表
    :param name: 问题名称
    :param num_tasks: 任务数量
    :param testName: 测试机名称
    """
    output_path = f"{output_dir}/{labs[labs_idx]}{sub_exp_name}/{testName}_{int(Problem.maxFE / (10000 * Problem.T))}"
    osi.makedirs(output_path, exist_ok=True)
    output_path = f"{output_path}/{name}.xlsx"

    if not isUpdateExcel:
        print(f"{name} update failed!")
        return

    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        for t in range(num_tasks):
            df = pd.DataFrame(T[t].T)
            df.to_excel(writer, sheet_name=f'T{t + 1}', index=False)

        workbook = writer.book
        format_excel_sheets(workbook)

    print(f"{name} update completed! File has been saved to: {output_path}")


def format_excel_sheets(workbook):
    """
    为Excel工作表应用格式化样式。

    :param workbook: openpyxl工作簿对象
    """
    border = Border(
        left=Side(style='thin'), right=Side(style='thin'),
        top=Side(style='thin'), bottom=Side(style='thin')
    )
    title_font = Font(name='Times New Roman', bold=True)
    cell_font = Font(name='Times New Roman')

    for sheet_name in workbook.sheetnames:
        worksheet = workbook[sheet_name]
        for row in worksheet.rows:
            for cell in row:
                cell.border = border
                cell.alignment = Alignment(horizontal='center', vertical='center')
                cell.font = title_font if cell.row == 1 else cell_font
                cell.number_format = '0' if cell.row == 1 else '0.00E+00'


def main():
    """
    主函数，协调优化流程。
    """
    parser = argparse.ArgumentParser(description="运行指定多任务优化问题（支持维度设置）")
    parser.add_argument('--func', type=int, required=True, help='功能编号（0-59）')
    parser.add_argument('--dim', type=int, default=None, help='维度设置（用于 PKACP 等问题；不传则使用默认值 50）')
    args = parser.parse_args()

    Prob, Name, testName = select_problem(args.func, args.dim)

    T = run_algorithm(Prob, Name)
    T = extract_continuous_best(T, task_idx)
    T = compute_average_convergence(T)
    save_to_excel(T, Name, Problem.T, testName)


if __name__ == "__main__":
    main()
