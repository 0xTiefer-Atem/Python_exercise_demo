# import导入模块的基本使用
import numpy as np
"""
首次导入会发生3件事
    1、执行numpy.py
    2、产生numpy.py的命名空间, 将numpy.py运行过程中产生的名字都丢到numpy的名称空间中
    3、在当前文件中产生的有一个名字numpy, 该名字指向2中产生的名称空间
    之后的的导入, 都是直接引用首次导入产生的numpy.py名称空间, 不会重复执行代码
"""

# 2、引用: 指名道姓的问某一个模块要名字对应的值
# 强调1: 模块名字.名字, 是指名道姓的问某一个模块要名字对应的值, 不会与当前名称空间中的名字发生冲突
# 强调2: 无论是查看还是修改都是以原模为基准的, 与调用模块无关


# 3、可以以逗号为分隔符在一行导入多个模块(但不建议)

# 4、导入模块规范
# 1)python内置模块
# 2)第三方模块
# 3)自定义模块

# 5、import ... as ...
# 给名字较长的模块器别名
import numpy as np

# 6、模块是第一类对象
# 7、自定义模块的命名应该采用纯小写+下划线的风格
# 8、可在函数内导入模块