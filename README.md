<<<<<<< HEAD
# Evolutionary Multitask Optimization

## 一、项目介绍

### 1. 项目结构

#### 1.1 算法部分

```text
|_Algorithms: 算法
    |_ManyTask: 超多任务算法
    |_MultiTask: 多任务算法
        |_Competitive: 竞争多任务优化算法
        |_MultiFactorial: 单种群优化算法
            |_MFEA: 多因子进化算法
            |_MFEA_AKT: MFEA知识迁移变体
            |_RLMFEA: 强化学习增强MFEA
        |_MultiPopulation: 多种群优化算法
			|_AEMTO: 自适应知识共享
            |_BLKT_DE: 块级知识迁移
			|_EMEA: 显式自动编码迁移
            |_MGTA: 基于偏差估计迁移
    |_SingleTask: 单任务算法
    |_Analyse: 数据处理(显著性分析+收敛图数据预处理)
    |_Run: 算法运行脚本
    |_Utils: 工具类
     	|_Individual: 个体类对象
    	|_MultiFactorial: 多因素算法的种群操作
        |_MultiPopulation: 多种群算法的种群操作
        |_Operator: 操作算子(DE、交叉算子、选择算子、变异算子)
    |_Algorithm: 算法基类(定义算法指标、评估、终止条件等)
```

#### 1.2 测试集部分
```text
|_Problems: 测试集
    |_ManyTask: 超多任务优化
    	|_CEC19-MaTSO
        |_WCCI20-MaTSO
    |_MultiTask: 多任务优化测试集
    	|_CEC17-MTSO
        |_WCCI20-MTSO
        |_Competitive-C2TOP
    |_RealWorld: 现实世界测试集
    	|_PKACP: 机械臂控制问题
        |_PEPVM: 光伏模型参数调整问题
    |_SingleTask: 单任务优化测试集
    	|_Classical-function: 古典函数测试集
    |_Base: 基准函数
    |_Problem: 测试集基类(定义问题指标、评估函数等)
```

### 3. 运行指南

#### 3.1 批量运行

1. 修改 `Run/run.sh` 中的测试集序列 `seq`
2. 修改 `Run/output.py`中的运行参数
3. 切换路径至 `Run`，运行指令`sh run.sh`

#### 3.2 单个运行

- 在算法文件中，选择指定的测试函数后，运行算法，如：`MFEA.py`
- 运行 `Run/output.py --func x`，形参 `x` 表示问题编号

#### 3.3 运行结果保存

若 `isUpdateExcel = True` ，每个问题的运行结果会独立保存至 `excel` 文件，路径为 `Files/算法类型/算法名称/实验名称/具体实验问题名称(可空)/测试集名称_评估次数(×万,每个任务的评估次数).xlsx`

## 二、项目编码规范

### 1. 新增算法和测试集

#### 1.1 新增算法模板

```python
class 算法名称(Algorithm):  # 必须继承 Algorithm 类
    def __init__(self):
        super().__init__()  # 调用父类的构造函数
        ……  # 超参数

    def run(self, Prob, isPrint=False):
        # 论文设定的每个子种群大小
        Problem.N =  
		……
        # 返回进化信息
        return self
```

#### 1.2 新增测试集模板

```python
class 测试集名称(Problem):  # 必须继承 Problem 类
    def __init__(self, dim= , lb= , ub= ):
        super().__init__(dim, lb, ub)  # 调用父类的构造函数
        ……  # 超参数

    def fnc(self, var):  # 必须实现 fnc 类，其中输入参数为种群基因型矩阵，输出为种群适应值列表
        ……
        return obj  # 形状为 (n,)
```

### 2. 编码规范

#### 2.1 提交代码前检查

在提交代码前，确保以下几点：

- 代码可正常运行，所有测试用例通过。
- 如果项目是复现论文，复现结果需确保与原论文中的结果一致。

#### 2.2 代码与论文信息

每个代码文件必须包含以下信息，并遵循格式要求：

```python
# <-*--*--*--*- Reference -*--*--*--*--*-> 
# @title: 论文全称 
# @Author: 论文作者 
# @Journal: 所属期刊 
# @year: 发表年份 
# @Doi: Doi号

# <-*--*--*--*- Coder -*--*--*--*--*-> 
# @Time: 编码时间 
# @Author: 编码人
# @Introduction: 文件简要介绍
# @Remind: 文件注意事项（如：复现结果与原论文差异等）

# <-*--*--*--*- Use -*--*--*--*--*->
使用指南
```

#### 2.3 编码注释

- 所有函数、类和复杂代码块必须有清晰的注释。
- 函数注释采用 `reStructuredText` 风格, 应详细描述功能、输入参数和返回值。函数注释模板如下：

```python
"""
功能介绍
:param 参数1: 参数1介绍
:param 参数2: 参数2介绍
:return: 
    返回值1: 返回值1介绍
    返回值2: 返回值2介绍
"""
```

#### 2.4 运行结果保存文件

- 运行结果保存为 Excel 文件时，**请不要将 Excel 文件上传至该仓库**。
- 仅上传代码和必要的文档。

## 许可证

该项目使用 [MIT 许可证](https://opensource.org/licenses/MIT)。

您可以自由地使用、修改、合并、发布、分发、再许可和/或销售该软件的副本，但必须遵守许可证的条件。
=======
# EMTO-Platform
EMTO-Platform is a Python research toolkit for evolutionary multitask optimization. It collects single-task, multitask, many-task, and real-world optimization problems together with representative evolutionary algorithms, so researchers can reproduce experiments, compare methods, and add new algorithms in a shared code structure.
>>>>>>> f9ce67b5f938d07db336da85954c3978995313a5
