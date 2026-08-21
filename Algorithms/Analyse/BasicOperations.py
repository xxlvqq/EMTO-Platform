# <-*--*--*--*- Coder -*--*--*--*--*->
# @Time: 2025/5/29 下午11:02
# @Author: wzb
# @Introduction: 基本操作类，提供数据选择、处理顺序获取和Excel格式化功能
# @Remind: 超参数和数据结构不要轻易更改

from openpyxl.styles import Font, Border, Side, Alignment


class BasicOperations:
    def __init__(self, project_root="EMTO"):
        self.project_root = project_root

        # 数据选项
        self.data_options = {
            "CEC": {
                "MTSO": ["CEC17_5", "CEC17_10", "CEC22_5", "CEC22_10"],
                "MaTSO": ["CEC19_5", "CEC19_10", "CEC20_5", "CEC20_10"]
            },
            "Real": {
                "PEPVM": ["PEPVM_5", "PEPVM_10"],
                "PKACP": ["PKACP_1", "PKACP_5"],
                "MRNP": ["MRNP_0"]
            }
        }

        # 文件处理顺序
        self.processing_orders = {
            "CEC/MTSO": {
                "CEC17": ['CI_HS', 'CI_MS', 'CI_LS', 'PI_HS', 'PI_MS', 'PI_LS', 'NI_HS', 'NI_MS', 'NI_LS'],
                "CEC22": ['Benchmark1', 'Benchmark2', 'Benchmark3', 'Benchmark4', 'Benchmark5', 'Benchmark6',
                          'Benchmark7', 'Benchmark8', 'Benchmark9', 'Benchmark10']
            },
            "CEC/MaTSO": {
                "CEC19": ['CEC19_MaTSO1', 'CEC19_MaTSO2', 'CEC19_MaTSO3', 'CEC19_MaTSO4', 'CEC19_MaTSO5',
                          'CEC19_MaTSO6'],
                "CEC20": ['WCCI20_MaTSO1', 'WCCI20_MaTSO2', 'WCCI20_MaTSO3', 'WCCI20_MaTSO4', 'WCCI20_MaTSO5',
                          'WCCI20_MaTSO6', 'WCCI20_MaTSO7', 'WCCI20_MaTSO8', 'WCCI20_MaTSO9', 'WCCI20_MaTSO10']
            },
            "Real/PEPVM": {
                "PEPVM": ['PEPVM']
            },
            "Real/PKACP": {
                "PKACP": ['PKACP']
            },
            "Real/MRNP": {
                "MRNP": ['MRNP1', 'MRNP2', 'MRNP3', 'MRNP4', 'MRNP5', 'MRNP6', 'MRNP7', 'MRNP8', 'MRNP9', 'MRNP10', 'MRNP11',
                        'MRNP12', 'MRNP13', 'MRNP14']
            }
        }

        # Excel格式化
        self.border = Border(left=Side(style='thin'), right=Side(style='thin'),
                             top=Side(style='thin'), bottom=Side(style='thin'))
        self.title_font = Font(name='Times New Roman', bold=True)
        self.cell_font = Font(name='Times New Roman')
        self.alignment = Alignment(horizontal='center', vertical='center')
        self.number_format = '0.00E+00'

    def select_data_source(self):
        """
        支持任意多级嵌套结构的数据源选择，输入0可回退到上一级选项。

        :return: tuple，(完整路径字符串(如CEC/MTSO), 选中的数据源(如CEC17_5、CEC22_10等)) 或 None（用户退出时）
        """
        history = []
        current = self.data_options
        while True:
            if isinstance(current, dict):
                options = list(current.keys())
                level_name = "/".join(history) if history else "数据类别"
                print(f"\n请选择 {level_name}：")
                for idx, opt in enumerate(options, 1):
                    print(f"{idx}. {opt}")
                print(f"请输入编号（1-{len(options)}），或回车默认：{options[0]}，输入0返回上一级")
                selected = self._select_option_with_back(options)
                if selected == "__back__":
                    if not history:
                        print("已退出选择。")
                        return None
                    history.pop()
                    # 回退到上一级
                    current = self.data_options
                    for h in history:
                        current = current[h]
                    continue
                history.append(selected)
                current = current[selected]
            else:
                # 到达叶子
                options = current
                level_name = "/".join(history)
                print(f"\n请选择 {level_name} 的数据来源：")
                for idx, opt in enumerate(options, 1):
                    print(f"{idx}. {opt}")
                print(f"请输入编号（1-{len(options)}），或回车默认：{options[0]}，输入0返回上一级")
                selected = self._select_option_with_back(options)
                if selected == "__back__":
                    history.pop()
                    # 回退到上一级
                    current = self.data_options
                    for h in history:
                        current = current[h]
                    continue
                return level_name, selected

    def _select_option_with_back(self, options):
        """
        通用选项选择器，支持输入0回退到上一级。

        :param options: list，当前可选项列表
        :return: str，选中的项或"__back__"表示回退
        """
        while True:
            choice = input().strip()
            if choice == "":
                return options[0]
            if choice == "0":
                return "__back__"
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(options):
                    return options[idx]
                print(f"无效输入，请输入 1-{len(options)} 的编号，或输入0返回上一级：")
            except ValueError:
                print(f"输入格式错误，请输入 1-{len(options)} 的编号，或输入0返回上一级：")

    def get_processing_order(self, category, data_source):
        """
        返回指定类别和数据源对应的处理顺序列表。

        :param category: str，数据类别（如"CEC/MTSO"）
        :param data_source: str，数据源名称
        :return: list，处理顺序
        :raises ValueError: 未找到对应处理顺序时抛出
        """
        for prefix, sources in self.processing_orders.items():
            if category == prefix:
                for key, order in sources.items():
                    if key in data_source:
                        return order
        raise ValueError(f"未定义处理顺序: {category}, {data_source}")

    def format_excel_file(self, writer):
        """
        应用Excel格式，包括边框、对齐、字体和数字格式。

        :param writer: openpyxl.Workbook，Excel写入对象
        """
        for sheet_name in writer.book.sheetnames:
            worksheet = writer.book[sheet_name]
            for row in worksheet.iter_rows():
                for cell in row:
                    cell.border = self.border
                    cell.alignment = self.alignment
                    cell.font = self.title_font if cell.row == 1 else self.cell_font
                    if cell.row != 1 and isinstance(cell.value, (int, float)):
                        cell.number_format = self.number_format
                    else:
                        cell.number_format = 'General'
