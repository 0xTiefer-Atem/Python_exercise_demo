"""
函数递归:
    是函数嵌套调用的一种特殊形式
    具体指的是:
        在调用函数的过程中有直接或者间接的调用了自己
"""

# 直接调用本身
# def f1(level):
#     print("f1 f1 f1 f1 " + str(level))
#     f1(level + 1)
#
#
# f1(1)


# 间接调用本身
# def f2():
#     print("===========> f1")
#     f3()
#
#
# def f3():
#     print("===========> f1")
#     f2()
#
#
# f2()

# 递归本质就是循环
# 强调:
# 递归不应该无限的调用下去, 必须在满足某种条件下结束递归


# 递归的两个阶段
# 回溯
# 递推

# 递归的应用
# 打印列表的值
l = [1, 2, [3, [4, [5, [6, [7, [8, [9, 10, 11]]]]]]]]


def print_element(l):
    for element in l:
        if type(element) is list:
            print_element(element)
        else:
            print(element)


print_element(l)
