# 作用域
# 全局作用域: 内置名称空间、全局名称空间
# 1、全局存活
# 2、全局有效: 被所有函数共享


# 局部作用域: 局部名称空间的名字
# 1、临时存活
# 2、局部有效: 函数内有效

# LEGB
"""
变量访问:
    L-Local(function)；函数内的名字空间
    E-Enclosing function locals；外部嵌套函数的名字空间(例如closure)
    G-Global(module)；函数定义所在模块（文件）的名字空间
    B-Builtin(Python)；Python内置模块的名字空间
当Python访问变量值时，默认LEGB查找原则，如果都找不到，则会抛出NameError

"""