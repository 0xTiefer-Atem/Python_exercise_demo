# p392 __init__方法
# 定义一个类
class Student:
    # 1、变量的定义
    stu_school = 'liao shi hua'

    # 2、定义初始化方法
    def __init__(obj, x, y, z):
        print(obj)
        obj.name = x
        obj.age = y
        obj.gender = z
        print('====>')

    # 3、功能的定义
    def tell_stu_info(stu_obj):
        print(stu_obj)

    def set_info(stu_obj):
        print(stu_obj, 'set_info')


# 调用类的过程又称之为实例化,发生了三件事
# 1、先产生一个空对象
# 2、调用类中的 __init__ 方法,然后将空对象以及调用 __init__ 时所传的参数一同传给 __init__ 方法
# 3、返回初始化好的对象
stu_obj = Student('wq', 18, 'male')
print(stu_obj)

# __init__ 方法使用总结
# 1、会在调用类时自动触发执行, 用来为对象初始化自己独有的数据
# 2、__init__ 内应该放的是为对象初始化属性的功能，但是可以写其他代码，
# 想要在类调用时就立刻执行的代码又可以放在 __init__ 方法内
# 3、__init__ 方法必须返回None