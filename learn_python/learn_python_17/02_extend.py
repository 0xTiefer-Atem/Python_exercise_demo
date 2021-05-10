# p372 继承
"""
什么是继承

    继承是一种创建新类的方式，新建的类可以继承一个或多个父类（python支持多继承），父类又可称为基类或超类，新建的类称为派生类或子类。

    子类会“”遗传”父类的属性，从而解决代码重用问题（比如练习7中Garen与Riven类有很多冗余的代码）
    python支持多继承
    在python中，新建的类可以继承一个或者多个父类
"""


class Parent1:
    pass


class Parent2:
    pass


# 单继承
class Sub1(Parent1):
    pass


# 多继承
class Sub2(Parent1, Parent2):
    pass


print(Sub1.__bases__)  # (<class '__main__.Parent1'>,)
print(Sub2.__bases__)  # (<class '__main__.Parent1'>, <class '__main__.Parent2'>)

# ps: 在python2中有经典类和新式类之分
# 新式类: 继承了object类的子类以及该子类的子类
# 经典类: 没有继承object的类及其子类

# ps: 在python3中没有继承任何类，默认继承object类
print(Parent1.__bases__)  # (<class 'object'>,)
print(Parent2.__bases__)


# 继承是解决类与类之间代码冗余的问题
