"""
闭包函数=名称空间与作用域+函数嵌套+函数对象
核心点：名字的查找关系是以函数定义阶段为准
"""

# 闭包函数
# "闭"函数指的是该函数是内嵌函数
# "包"函数指的是该函数包含对外层函数作用域名字的引用(不是全局作用域)

# 闭包函数: 名称空间与作用域的应用+函数嵌套
# def f1():
#     x = 1
#
#     # 闭的提现
#     def f2():
#         print(x)  # 包的提现
#     f2()


# 闭包函数: 函数对象
# 无论在哪里调用f2函数, f2内所需要的变量值在定义阶段就已经确定
# def f1():
#     x = 1
#
#     # 闭的提现
#     def f2():
#         print(x)  # 包的提现
#
#     return f2


# f = f1()
# print(f)  # <function f1.<locals>.f2 at 0x01A387C0>
# f()

# 为何使用闭包函数 应用场景
# 两种为函数体传参的方式

# 方式一: 直接把函数体需要的的参数定义成形参
# def f1(x):
#     print(x)


# 方式2:
# def f1():
#     x = 3
#
#     def f2():
#         print(x)


import requests


def outter(url):
    def get():
        response = requests.get(url)
        print(len(response.text))

    return get


f = outter("https://www.baidu.com")
f()
# get("https://www.baidu.com")
# get("http://47.107.64.157:8090")
