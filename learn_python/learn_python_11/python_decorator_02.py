# 装饰器
"""
1、定义
    器指的是工具,可以定义成函数
    装饰指的是为其他事物添加额外的功能点缀

    合到一起的解释:
        装饰器指的是定义一个函数(或者类), 该函数是用来装饰其他函数的, 即为其他函数增加功能的

2、为何要用装饰器
    开放封闭原则
        开放: 指的是对拓展功能是开放的
        封闭: 指的是对修改源代码是封闭的
    装饰器就是在不修改被装饰器对象源代码以及调用方式的前提下为被装饰的对象添加新功能
"""

# 使用
import time


# 需求 在不修改index函数源代码的情况下以及调用方式的前提下为其添加统计运行时间的功能
# def index(x, y):
#     print("index {}, {}".format(x, y))
#
#
# index(111, 222)

# 解决方案一:
# 问题: 修改了源代码, 违反了开放封闭原则
# def index(x, y):
#     start = time.time()
#     time.sleep(3)
#     print("index {}, {}".format(x, y))
#     end = time.time()
#     print(end - start)
#
#
# index(111, 222)


# 解决方案二
# 问题: 没有修改被装饰对象的调用方式, 也没有修改源代码, 并且加上了新功能
#      但是代码冗余
# def index(x, y):
#     time.sleep(3)
#     print("index {}, {}".format(x, y))
#
#
# start = time.time()
# index(111, 222)
# end = time.time()
# print(end - start)

# 解决方案三
# 问题: 解决了方案二代码冗余问题, 但带来一个新问题就是函数的调用方式发生了改变
# def index(x, y):
#     time.sleep(3)
#     print("index {}, {}".format(x, y))
#
#
# def wrapper():
#     start = time.time()
#     index(111, 222)
#     end = time.time()
#     print(end - start)
#
#
# wrapper()

# 方案三优化一: 将index参数写活
# def index(x, y):
#     time.sleep(3)
#     print("index {}, {}".format(x, y))
#
#
# def wrapper(*args, **kwargs):
#     start = time.time()
#     index(*args, **kwargs)
#     end = time.time()
#     print(end - start)
#
#
# wrapper(111, y="dfd")


# 方案三-优化二: 在优化一的基础上把被装饰对象写活了, 原来只能装饰index
# def index(x, y, z):
#     time.sleep(3)
#     print("index {}, {}, {}".format(x, y, z))
#
#
# print(index)
#
#
# def outer(func):
#     def wrapper(*args, **kwargs):
#         start = time.time()
#         func(*args, **kwargs)
#         end = time.time()
#         print(end - start)
#
#     return wrapper
#
#
# 前一个index指向的是outer里的wrapper函数的内存地址, 后一个index还是原来的index函数的内存地址
# index = outer(index)
# print(index)
# index(111, 222, z=333)


# 方案三-优化三: 将wrapper做的跟被装饰对象一模一样
# def index(x, y, z):
#     time.sleep(3)
#     print("index {}, {}, {}".format(x, y, z))
#
#
# def home(name):
#     time.sleep(2)
#     print("welcome {} to home page".format(name))
#     return 123
#
#
# print(index)
#
#
# def outer(func):
#     def wrapper(*args, **kwargs):
#         start = time.time()
#         res = func(*args, **kwargs)
#         end = time.time()
#         print(end - start)
#         return res
#
#     return wrapper
#
#
# home = outer(home)
# res = home("asd")
# print("返回值-->", res)
# index = outer(index)
# print(index)
# index(111, 222, z=333)


# 解决方案四: 如何在方案三的基础上不改变函数的调用方式


# 语法糖: 在被装饰对象头上添加注解
# 装饰器
# def timmer(func):
#     def wrapper(*args, **kwargs):
#         start = time.time()
#         res = func(*args, **kwargs)
#         end = time.time()
#         print(end - start)
#         return res
#
#     return wrapper
#
#
# @timmer
# def index(x, y, z):
#     time.sleep(3)
#     print("index {}, {}, {}".format(x, y, z))
#
#
# @timmer
# def home(name):
#     time.sleep(2)
#     print("welcome {} to home page".format(name))
#     return 123
#
#
# res = home("asd")
# print("返回值-->", res)


# 总结 无参装饰器模板
def outer(func):
    def wrapper(*args, **kwargs):
        # 1、增加新功能
        # 2、调用原函数
        res = func(*args, **kwargs)
        return res

    return wrapper