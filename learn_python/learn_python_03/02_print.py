# 与用户交互
# 输入语句
# 在python3中，input方法会将所有输入的内容转换成字符串
# username = input("请输入用户名: ")
# print(username, type(username))

# 类型转换
# print("请输入年龄: ")
# age = input()
# age = int(age)  # int函数只能将纯数字的字符串转成整型，不包括小数的字符串形式
# print(age, type(age))
# x = float("12.5")
# print(x, type(x))
# l = list("[1,2,3]")
# y = type(l)
# print(l, type(y))


# 格式化字符串输出
# 方式一 %形式 %s可以接收任意类型  %d只能接收数字类型
# 左面百分号的位置与右侧括号里的位置时一
# 一对应的，绝对不能少，也不能多，否则都活报错
# res = "my name is %s, my age is %s" % ("wq", "18")
# print(res)

# res = "我的名字是 %(name)s, 我的年龄是 %(age)s" % {"name": "wq", "age": "18"}
# print(res)


# 方式二 str.format 推荐使用，兼容性好
# 按照位置传
# res = "我的名字是 {}, 我的年龄是 {}".format('wq', 18)
# print(res)

# res = "我的名字是 {0}{0}, 我的年龄是 {1}".format('wq', 18)
# print(res)

# 打破位置传值
# res = "我的名字是 {name}{name}, 我的年龄是 {age}".format(name='wq', age=18)
# print(res)


# 方式三 f符号 3.5版本才推出
name = input("你的名字: ")
age = input("你的年龄: ")
res = f'你的名字: {name} 你的年龄{age}'
print(res)
