# p370 371 property使用
# 装饰器是在不修改被装饰对象源代码以及调用方式的前提下
# 为被装饰对象添加新功能的可调用对象

# property是一个装饰器,是用来绑定给对象的方法伪造成一个数据属性

# 一
# 计算体质参数
# class People:
#     def __init__(self, name, weight, height):
#         self.name = name
#         self.weight = weight
#         self.height = height
#
#     @property
#     def bmi(self):
#         return self.weight / (self.height * self.height)
#
#
# obj1 = People('xht', 43, 1.59)
# print(obj1.bmi)


# 二 老版本用法
# class People:
#     def __init__(self, name):
#         self.__name = name
#
#     def get_name(self):
#         return self.__name
#
#     def set_name(self, name):
#         if type(name) is not str:
#             print("必须传入str类型")
#             return
#         self.__name = name
#     name = property(get_name, set_name)


# obj = People('qwe')
# # print(obj.get_name())
# # obj.set_name('asd')
# # print(obj.get_name())
#
#
# print(obj.name)
# obj.name = 18
# print(obj.name)

# 三 新用法 效果跟上一个一样
class People:
    def __init__(self, name):
        self.__name = name

    @property
    def name(self):
        return self.__name

    @name.setter
    def name(self, name):
        if type(name) is not str:
            print("必须传入str类型")
            return
        self.__name = name

    @name.deleter
    def name(self):
        print("不能删除")


obj = People('qwe')

print(obj.name)
obj.name = 18
print(obj.name)
