# 1、编写有参装饰器
# def deco(user, pwd):
#     def outer(func):
#         def wrapper(*args, **kwargs):
#             print("invoke wrapper!")
#             input_user = input("input user: ")
#             input_pwd = input("input user: ")
#             if input_user == user and input_pwd == pwd:
#                 res = func(*args, **kwargs)
#                 return res
#             else:
#                 print("error!")
#
#         return wrapper
#
#     return outer
#
#
# @deco(user="wq", pwd="123")
# def index(name):
#     print("hello {} !".format(name))
#
#
# index("wq")


# 2、文件开头声明一个字典, 然后再每个函数上加上装饰器注解
# 完成自动添加进字典的操作
fun_dic = {}


def deco(index):
    def outer(func):
        fun_dic[index] = func

        def wrapper(*args, **kwargs):
            res = func(*args, **kwargs)
            return res

        return wrapper

    return outer


@deco(index="1")
def index(name):
    print("index {}".format(name))


@deco(index='2')
def login():
    print("login !")


@deco(index='3')
def out():
    print("out !")


print(fun_dic)
