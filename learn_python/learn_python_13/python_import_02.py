# 循环导入问题

# 模块搜索优先级
# from foo import x
# import foo
# 无论是import ... 还是 from ... import ...再导入模块时都涉及到优先级问题
# 优先级:
# 1、内存(内置模块)
# 2、从硬盘找 按照sys.path 中存放的文件顺序依次查找要导入的模块

import sys

# 值为第一个列表, 存放了一系列文件夹
# 其中第一个文件夹是当前执行文件所在的文件夹
# print(sys.path)

# sys.modules查看已经加载到内存中的模块
# print(sys.modules)


# sys.path的应用
# sys.path.append()

# 编写规范的模块

# "The module is used to..."  # 模块的文档描述
#
# import sys  # 导入模块
#
# x = 1  # 定义全局变量,如果非必须,则最好使用局部变量,这样可以提高代码的易维护性,并且可以节省内存提高性能
#
# class Foo:  # 定义类,并写好类的注释
#     'Class Foo is used to...'
#     pass
#
#
# def test():  # 定义函数,并写好函数的注释
#     'Function test is used to…'
#     pass
#
#
# if __name__ == '__main__':  # 主程序
#     test()  # 在被当做脚本执行时,执行此处的代码
