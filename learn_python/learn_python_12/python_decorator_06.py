# 一、叠加多个装饰器的加载、运行分析(了解***)

# 装饰器一
def deco1(func1):  # func1 = wrapper2的内存地址
    # print(func1)

    def wrapper1(*args, **kwargs):
        print("deco1.wrapper1.invoke")
        res1 = func1(*args, **kwargs)
        return res1

    return wrapper1


# 装饰器二
def deco2(func2):  # func2 = wrapper3的内存地址
    # print(func2)

    def wrapper2(*args, **kwargs):
        print("deco2.wrapper2.invoke")
        res2 = func2(*args, **kwargs)
        return res2

    return wrapper2


# 装饰器三
def deco3(x):
    def outer3(func3):  # func3 = 被装饰对象index函数的内存地址
        # print(func3)

        def wrapper3(*args, **kwargs):
            print("deco3.outer3.wrapper3.invoke    {}".format(x))
            res3 = func3(*args, **kwargs)
            return res3

        return wrapper3

    return outer3


"""
1、加载顺序自下而上
"""


@deco1  # index = deco1(wrapper2的内存地址) -->  index = wrapper1的内存地址
@deco2  # index = deco2(wrapper3的内存地址) -->  index = wrapper2的内存地址
@deco3("deco3")  # --> @outer3 --> index = outer3(index)  --> index = wrapper3的内存地址
def index(x, y):
    print("from index: {}, {}".format(x, y))


# print(index)

"""
2、执行顺序: 自上而下, 即 wrapper1-->wrapper2-->wrapper3
"""
index(1, 3)  # wrapper1(1, 3)
