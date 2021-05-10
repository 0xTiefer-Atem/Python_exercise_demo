# 运算符
# 1. 算数运算符
# print(10 + 9)
# print(10 / 3)  # 结果带小数
# print(10 // 3)  # 结果只有整数，不会四舍五入
# print(10 % 3)  # 取余数
# print(10 ** 3)  # 表示乘方

# 2. 比较运算符

# 3.赋值运算符
# =: 变量赋值
# 增量赋值
# age = 19
# age += 1
# print(age)

# 链式赋值
# x = y = z = 10
# print(id(x), id(y), id(z))

# 交叉赋值
# m = 10
# n = 20
# s = 30
# print(n, m)
# m, n, s = s, n, m
# print(n, m, s)

# 解压赋值
# salaries = [111, 222, 333, 444]
# mo1, mo2, mo3, mo4 = salaries  # 最好 一个不少，一个不多
# print(mo1, mo2, mo3, mo4)

# 取前三个值
# x, y, z, *_ = salaries  # 会将没有对应关系的值存成列表然后赋值给紧跟*后的变量，此处的变量名为
# print(_)

# 取后三个值
# *_, a, b, c = salaries
# print(_)

# 取头和尾
# d, *_, e = salaries
# print(_)

# 就是不能取中间的值


# 解压字典默认解压出来的是字典里的key
# x, y, z = dic = {'a': 1, 'b': 2, 'c': 3}
# print(x, y, z)


