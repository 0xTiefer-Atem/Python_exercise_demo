# global 与 nonlocal

# 示例一: 针对不可变类型的修改, 若想局部修改全局的变量 先加global关键字声明
# x = 111
#
#
# def func():
#     #
#     global x
#     x = 222
#
#
# func()
# print(x)


# 示例二
# l = [111, 222, 333]
#
#
# def func():
#     l.append(3)
#
#
# func()
# print(l)

# nonlocal(了解): 修改函数外层函数包含的名字对应的值(不可变类型)
x = 0


def f1():
    x = 1

    def f2():
        nonlocal x
        x = 2

    f2()
    print('f1函数内的x: ', x)


f1()
print(x)
