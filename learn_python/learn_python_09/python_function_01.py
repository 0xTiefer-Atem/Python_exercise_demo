# 函数
'''

使用
   先定义
        三种定义方式
   后使用
        三种调用方式
   返回值
        三种返回值形式

def 函数名(参数1,参数2,...):
    """文档描述"""
    函数体
    return 值

'''


# 1、定义
# 形式一无参函数
# def func():
#     print("name")
#
#
# print(func)
# func()

# 定义函数发生的事情
# 1、申请内存空间
# 2、将上述内存地址绑定函数名
# 3、定义函数不会执行函数体代码，但是会检测函数体语法


# 调用函数发生的事情
# 1、通过函数名，找到函数的内存地址
# 2、加括号就是在触发函数体的执行


# 示范1
# x = 100
#
#
# def bar():
#     print("bar")
#
#
# def foo():
#     bar()
#     print(x)
#     print("哈哈哈")
#
#
# foo()

# 示范2
# 示范1
# x = 100
#
#
# def foo():
#     bar()
#     print(x)
#     print("哈哈哈")
#
#
# def bar():
#     print("bar")
#
#
# foo()


# 形式2: 有参函数
def my_min(x, y):
    if x > y:
        print(x)
    else:
        print(y)


my_min(1, 2)


# 形式三: 空函数，函数体代码为pass
def func(x, y):  # 参数可有可没有
    pass


# 三种定义方式各用在何处
# 1、无参函数的应用场景
# 不需要传参
# 2、有参函数的应用场景
# 需要传参
# 3、空函数的应用场景


# 二、调用函数
# 1、语句形式：只调用函数，没有任何其他操作
# add()

# 2、表达式形式:
# res = add(1,2
#
#
# 3、函数调用可当作参数
# add(4,add(1,2))


# 三、函数返回值
# return
# return是函数的结束标志，即函数体代码一旦运行到return会立刻终止函数的运行
# 并且会将return后的值当作本次运行的返回结果
# 1、函数内没有return、return空值、return None
# 2、返回一个值: return 值


# 3、返回多个值
# def return_mul_value():
#     return 1, "哈哈", [1, 2, "qw"]
#
#
# res = return_mul_value()
# print(res, type(res))  # (1, '哈哈', [1, 2, 'qw']) <class 'tuple'> 返回值是个元组

