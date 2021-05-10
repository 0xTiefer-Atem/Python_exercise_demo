# p388 面向对象编程思想
"""
面向过程:
        核心是"过程"
        核心思想就是将程序流程化
        过程是"流水线", 用来分步解决问题

面向对象:
        核心是"对象"
        对象的核心思想就是将程序"整合"
        对象是"容器", 用来放数据与功能的
"""


# p389 类与对象的介绍


# p390 类的定义

# 先定义类
# 类是对象相似数据与功能的集合体
# 所以类中最常见的是变量与函数的定义，但是类体中其实是可以包含任意其他代码的
# 注意：类体代码是在类定义阶段就会立即执行，会产生类的名称空间
class Student:
    # 1、变量的定义
    stu_school = 'liao shi hua'

    # 2、功能的定义
    def tell_stu_info(stu_obj):
        print(stu_obj)

    def set_info(stu_obj):
        print(stu_obj, 'set_info')

    # print("======>")


# 访问数据属性
# print(Student.__dict__['stu_school'])
# print(Student.stu_school)  # 本质就是 Student.__dict__['stu_school']


# 访问函数属性
# print(Student.set_info)  # <function Student.set_info at 0x015FC388>


# p391 创造对象
stu_obj1 = Student()
stu_obj2 = Student()
stu_obj3 = Student()


stu_obj1.name = 'wq'  # stu_obj1.__dict__['name'] = 'wq'
stu_obj1.age = 18  # stu_obj1.__dict__['age'] = 18
stu_obj1.gender = 'male'  # stu_obj1.__dict__['gender'] = 'male'
print(stu_obj1.__dict__)
