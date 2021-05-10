"""
自定义迭代器: 生成器
"""


# 如何的到自定义迭代器
# 在函数内, 一但存在yield关键字, 调用函数并不会执行函数体代码
# 会返回一个生成器对象, 生成器及自定义的迭代器
# def func():
#     print("第一次")
#     yield 1
#     print("第二次")
#     yield 2
#     print("第三次")
#     yield 3
#     print("第四次")
#     yield 4


# g = func()
# print(g)  # <generator object func at 0x0BA48CD8>
# 生成器就是迭代器
# res1.__iter__()
# res1.__next__()

# 会触发函数体代码的运行, 遇到yield停下来
# 将yield后的值当作本次调用的结果返回
# res1 = g.__next__()  # 打印的内容: 第一次
# print(res1)  # 1


# 应用案例
# 需求: 编写一个可产生大量个数值的迭代器
# def my_range(start, stop, step=1):
#     print("start...")
#     while start < stop:
#         yield start
#         start += step
#
#     print("end...")


# g = my_range(1, 5, 2)
# print(next(g))
# print(next(g))
# print(next(g))

# 将自定义的range函数使用到佛如循环里面
# for n in my_range(1, 7, ):
#     print(n)

