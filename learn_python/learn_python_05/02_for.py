# for循环
# 1、基本使用

'''
语法：
for 变量名 in 可迭代对象: 可迭代对象可以是: 列表，字典，字符串，元组，集合
    代码1
    代码2
    代码3
    ...
'''
# 例一: 列表循环取值
# l = [1, 2, 3, 4, 5]
# for x in l:
#     print(x, end=" ")


# 例二: 字典循环取值
# dics = {"k1": 1, "k2": 2, "k3": 3}
# for dic in dics:
#     print(dic, dics[dic], end="     ")

# 例三: 字符串循环取值
# strs = "a b bc d g t r d f"
# for str in strs:
#     print(str, end="     ")


# 小结
# 1、相同之处: 都是循环，for可以做的事，while也可以做
# 2、不同之处:
#            while称之为条件循环，循环次数取决于条件何时变为假
#            for成为取值循环，循环次数取决于in后面包含值的个数


# 利用for控制循环次数： 利用range() 函数
# range(10)  # 自动生成0-9的列表
# range(1, 9)  # 生成1-8的列表
# range(1, 9, 2)  # 生成1-9的列表，不过自增为2

# for x in range(9):
#     print("asdasda",x)


# for + break + else 同while一样

# user_name = "wq"
# user_pwd = "123"
#
# for x in range(3):
#     input_name = input("请输入用户名: ")
#     input_pwd = input("请输入密码: ")
#     if input_name == user_name and input_pwd == user_pwd:
#         print("登陆成功")
#         break
# else:
#     print("输错账号或者密码次数过多")


# 补充  使用len函数和range函数一起for循环
# l = [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
# for i in range(len(l)):
#     print(i)


# for + continue
# for i in range(10):
#     if i % 2 == 0:
#         continue
#     print(i)
