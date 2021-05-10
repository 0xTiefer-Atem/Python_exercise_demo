# from import 模块导入

# import导入模块在使用的时候的优缺点
# 优点: 肯定不会与当前的名称空间冲突
# 缺点: 加前缀显得麻烦

# from ... import ... 导入发生了三件事
# 1、产生一个模块的名称空间
# 2、运行foo.py将运行过程中产生的名字都丢到模块的名称空间去
# 3、在当前名称空间拿到一个名字，该名字与模块名称空间中的某一个内存地址

# from foo import x x = 模块foo中值的内存地址
# from foo import change
# from foo import get

# from ... import ... 导入模块在使用时不用加前缀
# 优点: 代码更精简
# 缺点: 容易与当前名称空间混淆

# 补充
# 1、from 这种方式也可以在一行导入多个名字(不推荐)

# 2、from foo import *
# *的用途: 导入模块中的所有名字(不推荐)

# 了解: __all__

# 起别名