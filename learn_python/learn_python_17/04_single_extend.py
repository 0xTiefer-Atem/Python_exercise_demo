# p374 单继承背景下的属性查找

# 示范一
# class Foo:
#     def f1(self):
#         print('from Foo.f1')
#
#     def f2(self):
#         print('from Foo.f2')
#         self.f1()  # 此时self = obj 所以调用的是obj的f1也就是调用的Bar里的f1
#
#
# class Bar(Foo):
#     def f1(self):
#         print('from Bar.f1')
#
#
# obj = Bar()
# obj.f2()



# 示范二   子类调用父类的方法
# class Foo:
#     def f1(self):
#         print('from Foo.f1')
#
#     def f2(self):
#         print('from Foo.f2')
#         Foo.f1(self)
#
#
# class Bar(Foo):
#     def f1(self):
#         print('from Bar.f1')
#
#
# obj = Bar()
# obj.f2()


# 示范三
# class Foo:
#     def __f1(self):
#         print('from Foo.f1')
#
#     def f2(self):
#         print('from Foo.f2')
#         self.__f1()  # 调用的是 self._Foo__f1
#
# class Bar(Foo):
#     def f1(self):
#         print('from Bar.f1')
#
#
# obj = Bar()
# obj.f2()