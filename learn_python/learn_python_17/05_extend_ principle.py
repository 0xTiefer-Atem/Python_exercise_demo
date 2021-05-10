# p375 继承的实现原理

class A(object):
    def test(self):
        print('from A')


class B(A):
    def test(self):
        print('from B')


class C(A):
    def test(self):
        print('from C')


class D(B, C):
    pass


obj = D()
print(D.mro())  # 类以及该类的对象访问属性都是参照该类的mro列表
# obj.test()  # 结果为：from B

# 对于你定义的每一个类，Python都会计算出一个方法解析顺序(MRO)列表，该MRO列表就是一个简单的所有基类的线性顺序列表

# 总结: 类相关的属性查找(类.属性、该类的对象.属性)，都参照该类的mro
"""
python会在MRO列表上从左到右开始查找基类,直到找到第一个匹配这个属性的类为止。 
而这个MRO列表的构造是通过一个C3线性化算法来实现的。我们不去深究这个算法的数学原理,
它实际上就是合并所有父类的MRO列表并遵循如下三条准则:

1.子类会先于父类被检查
2.多个父类会根据它们在列表中的顺序被检查
3.如果对下一个类存在两个合法的选择,选择第一个父类
所以obj.test()的查找顺序是，先从对象obj本身的属性里找方法test，没有找到，
则参照属性查找的发起者(即obj)所处类D的MRO列表来依次检索，首先在类D中未找到，
然后再B中找到方法test
"""

# 二  如果多继承是非菱形继承下，python2与python3的属性查找顺序一样
#       都是一个分支一个分支的找下去, 最后找到object

#  举例
class E:
    def test(self):
        print('from E')


class F:
    def test(self):
        print('from F')


class B(E):
    def test(self):
        print('from B')


class C(F):
    def test(self):
        print('from C')


class D:
    def test(self):
        print('from D')


class A(B, C, D):
    # def test(self):
    #     print('from A')
    pass


print(A.mro())
'''
[<class '__main__.A'>, <class '__main__.B'>, <class '__main__.E'>, <class '__main__.C'>, <class '__main__.F'>, <class '__main__.D'>, <class 'object'>]
'''

obj = A()
obj.test() # 结果为：from B
# 可依次注释上述类中的方法test来进行验证
