import time

# 1、编写一个函数
# def index(name):
#     print("欢迎 {} 来到主页".format(name))


# 2、编写装饰器, 为函数加上统计时间功能

# def time_counter(func):
#     def wrapper(*args, **kwargs):
#         start = time.time()
#         res = func(*args, **kwargs)
#         end = time.time()
#         print("当前函数 {} 执行时间为: {} s".format(func.__name__, end - start))
#         return res
#
#     return wrapper
#
#
# @time_counter
# def index(name):
#     time.sleep(2)
#     print("欢迎 {} 来到主页".format(name))
#
#
# index("wq")

# 3、编写装饰器, 为函数加上认证功能
# def login():
#     user_name = input("请输入账户:  ")
#     user_pwd = input("请输入密码:  ")
#     with open('./doc/account.txt', mode='rt', encoding="utf-8") as account_file:
#         msg = account_file.readline()
#         userinfo = eval(msg)
#     if userinfo['user'] == user_name:
#         if userinfo['password'] == user_pwd:
#             return True
#         else:
#             print("密码错误")
#             return False
#     else:
#         print("账户错误")
#         return False
#
#
# def auth(func):
#     def wrapper(*args, **kwargs):
#         res = login()
#         if res:
#             print("登录成功!")
#             func(*args, **kwargs)
#         else:
#             print("登陆失败, 请重新登录!")
#
#     return wrapper
#
#
# @auth
# def index(name):
#     print("欢迎 {} 来到主页".format(name))
#
#
# index("wq")


# 4、编写装饰器, 为多个函数加上认证功能, 要求只需登录成功一次, 后续的函数都无需再输入用户名和密码
# 记录成功登录的用户信息
auth_record = []


def login():
    user_name = input("请输入账户:  ")

    with open('./doc/account.txt', mode='rt', encoding="utf-8") as account_file:
        msg = account_file.readline()
        userinfo = eval(msg)
    if user_name in auth_record:
        print("有该用户登陆记录, 不用再次登录")
        return True
    user_pwd = input("请输入密码:  ")
    if userinfo['user'] == user_name:
        if userinfo['password'] == user_pwd:
            auth_record.append(user_name)
            return True
        else:
            print("密码错误")
            return False
    else:
        print("账户错误")
        return False


def auth(func):
    def wrapper(*args, **kwargs):
        while True:
            res = login()
            if res:
                print("登录成功!")
                func(*args, **kwargs)
            else:
                print("登陆失败, 请重新登录!")

    return wrapper


@auth
def index(name):
    print("欢迎 {} 来到主页".format(name))


index("wq")
