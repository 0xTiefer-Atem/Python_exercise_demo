# 变量 记录事物的状态
# 1、 变量的基本使用
# 原则：先定义后使用

# name = 'wq'  # 定义变量
# print(name)  # 变量的使用

# 2、内存管理：内存回收机制 垃圾指的是变量值被绑定的变量名的个数为0时，该变量值无法被访问到，称之为垃圾
# 引用计数法
# 引用计数增加
x = 3
y = x
c = x

# 引用计数减少
del x  # 解除变量名x与值10的绑定关系
del y  # 解除变量名y与值10的绑定关系
c = 1234
# print(c)

# 变量值三个重要的特征
name = 'wq'
# id：反映的是变量值的内存地址，内存地址不同则id不同
# print(id(name))  # 查看name的id号
# type：不同类型的值用来表示记录不同的状态
# print(type(name))  # 查看变量类型
# value：值本身
# print(name)  # 查看值本身


# id不同的情况下，只有可能相同，即两块不同的内存空间里也可以存相同的值
# id相同的情况下，值一定相同，x is y成立，x == y 必然也成立
# is与==
x = 'aaaa'
y = 'aaaa'
# is比较左右两个变量地址是是否相等
print(x is y)
# ==比较左右两个变量的值是否相等
print(x == y)


# 小整数池 范围[-5,256]
# 从python解释器启动那一刻开始，就会在内存中事先申请好一系列内存空间存放好常用整数


# 常量：不变的量
# python中没有常量的概念，但在程序开发的过程中会涉及到常量的概念
AGE_OF_XXX = 100  # 小写字母全为大写代表常量，只是一种约定，没有实际的关键字
# 第62个视频
