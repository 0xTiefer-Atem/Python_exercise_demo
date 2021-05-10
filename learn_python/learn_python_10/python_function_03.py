# 函数嵌套
# 1、函数嵌套的调用: 在调用一个函数的过程又调用其他函数
def max2(x, y):
    if x > y:
        return x
    else:
        return y


def max4(a, b, c, d):
    res = max2(a, b)
    res1 = max2(c, d)
    res3 = max2(res1, res)
    return res3


# 函数的嵌套定义: 在函数内定义函数
# def f1():
#     def f2():
#         pass

# 圆形
from math import pi


# 周长
def circle(radius, action=0):
    def perimiter(radius):
        return 2 * pi * radius

    # 面积
    def area(radius):
        return pi * radius ** 2

    if action == 0:
        return perimiter(radius)
    else:
        return area(radius)
