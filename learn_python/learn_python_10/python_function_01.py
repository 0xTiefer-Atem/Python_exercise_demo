# 函数对象
# 精髓: 可以把函数当成变量去用
# fun = 地址
def func():
    print("from func")


# 1、可以赋值
# f = func
# print(f, func)
# f()


# 2、可以当作函数的参数传给另外一个函数
# def foo(x):
#     print(x)
#     x()
#
#
# foo(func)  # foo(func的内存地址)

# 3、可以当作另外一个函数的返回值
# def foo():
#     return func
#
#
# x = foo()
# x()

# 4、可以当作容器类型的一个元素
# l = [func, ]
# print(l)
# l[0]()
#
# d = {'k1': func}
# print(d)


# 结合ATM取款机
def login():
    print("登录")


def transfer():
    print("转账功能")


def check_balance():
    print("查询余额")


def withdraw():
    print("提现")


fun_dic = {
    '1': login,
    '2': transfer,
    '3': check_balance,
    '4': withdraw
}

while True:
    print("""
    0 退出
    1 登录
    2 转账
    3 查询余额
    4 提现
    """)
    command = input("请输入命令: ")
    if not command.isdigit():
        print("必须是以上编号,憨批")
        continue
    if command == '0':
        break
    if command in fun_dic:
        fun_dic[command]()
    else:
        print("没有该命令")