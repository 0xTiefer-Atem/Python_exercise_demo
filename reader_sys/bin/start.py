"""
项目启动入口
"""
# 1、导入系统模块
import os
import sys
from core import src

# 2、将项目的根目录, 添加到sys.path中
sys.path.append(
    os.path.dirname(os.path.dirname(__file__))
)

if __name__ == '__main__':
    src.run()
