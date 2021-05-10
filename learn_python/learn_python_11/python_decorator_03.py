"""
有参装饰器
"""


# 一: 知识储备
# 由于语法糖的限制, outer函数只能有一个参数
# 而且该参数只用来接收被装饰对象的内存地址
# def outer(func):
#     def wrapper(*args, **kwargs):
#         res = func(*args, **kwargs)
#         return res
#     return wrapper
#
#
# @outer
# def index(name):
#     print("欢迎 {} ".format(name))
#
#
# index("wq")

# 有参装饰器
def auth(db_type):
    def deco(func):
        def wrapper(*args, **kwargs):
            name = input("your name: ")
            password = input("your password: ")
            if db_type == 'file':
                print("来自文件验证")
                if name == 'wq' and password == '123':
                    print("login successful")
                    res = func(*args, **kwargs)
                    return res
                else:
                    print("user or pass error")
            if db_type == 'mysql':
                print("来自数据库验证")
            if db_type == 'ldap':
                print("来自ldap验证")

        return wrapper

    return deco


@auth(db_type='file')
def index(name):
    print("index {} ".format(name))


@auth(db_type='mysql')
def home(name):
    print("home {} ".format(name))


@auth(db_type='ldap')
def login(name):
    print("login {} ".format(name))


index('wq')
home('wasdasd')
login('qweqweq')


# 有参装饰器模板
def deco_para(x, y, z):
    def outer(func):
        def wrapper(*args, **kwargs):
            res = func(*args, **kwargs)
            return res

        return wrapper

    return outer


@deco_para(1, 2, 3)
def target():
    pass
