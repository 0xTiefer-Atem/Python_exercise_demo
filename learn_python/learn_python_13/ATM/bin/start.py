# 启动文件

# 针对非包, 采用绝对导入
import sys
# sys.path.append(r'F:\Jetbrains\PythonSoftWare\PythonProjects\learn_python\learn_python_13\ATM')
# from core import src
# src.run()

# 导包的优化方案
import os

# 当前文件的绝对路径
# print(__file__)
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(BASE_DIR)
from core import src

if __name__ == '__main__':
    src.run()
