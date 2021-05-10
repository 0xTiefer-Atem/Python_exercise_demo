# 面向过程的编程思想
# 核心是"过程"二字,过程及流程,指的是做事的步骤: 先干什么 在干什么 后干什么
# 基于该思想编写程序 就好比在设计流水线
# 优点: 复杂的问题流程化、简单化
# 缺点: 扩展性非常差


# 函数式编程
# 1、有名函数
# func = 函数的内存地址
# def func(x, y):
#     return x + y


# print(func)  # <function func at 0x0214C028>

# 2、匿名函数: lambda 定义匿名函数
# 编写格式: lambda 参数1,参数2:表达式
# print(lambda x, y: x + y)  # <function <lambda> at 0x014F87C0>

# 3、调用匿名函数: 函数内存地址加()
# 方式一:
# res = (lambda x, y: x + y)(1, 2)
# print(res)

# 方式二:
# func1 = lambda x, y: x + y
# print(func1)  # <function <lambda> at 0x01DA87C0>
# res = func1(2, 3)
# print(res)

# 4、匿名函数用于临时调用一次的场景: 更多的是将匿名函数与其他函数配合的场景

# 5、匿名函数的应用

salaries = {
    'siry': 3000,
    'tom': 7000,
    'lili': 10000,
    'jack': 2000
}

# 需求1: 找出薪资最高的那个人
# 方法一:
# def return_value(k):
#     return salaries[k]
#
#
# res = max(salaries, key=return_value)
# print(res, salaries[res])


# 方法二
# res = max(salaries, key=lambda k: salaries[k])
# print(res, salaries[res])


# sorted方法应用
# 根据工资进行比较
# res = sorted(salaries, key=lambda k: salaries[k])
# print(res)

# map的应用
# 在每个元素后面添加'_xxx'
# l = ['alex', 'asda', 'qwe']

# 方案一: 列表生成式
# new_l = [name + '_xxx' for name in l]
# print(new_l)


# 方案二: map函数
# res = map(lambda name:name + '_xxx',l)
# print(res)


# filter的应用
# 需求选出后缀是_xx的字符串
l = ['alex_xx', 'asda', 'qwe']

# 方案一
# res = [name for name in l if name.endswith('_xx')]
# print(res)


# 方案二
# res = filter(lambda name: name.endswith('_xx'), l)
# print(res.__next__())

# reduce的应用
from functools import reduce

res = reduce(lambda x, y: x + y, [1, 2, 3, 4], 10)
print(res)


res1 = reduce(lambda x, y: x + y, ['a', 'q', 'w', 'e'])
print(res1)
