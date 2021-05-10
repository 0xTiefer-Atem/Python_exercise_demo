# 属性查找与绑定方法

class Student:
    # 1、变量的定义
    stu_school = 'liao shi hua'

    # 2、定义初始化方法
    def __init__(self, x, y, z):
        self.name = x
        self.age = y
        self.gender = z

    # 3、功能的定义
    def tell_stu_info(self):
        print('学生信息:[ 姓名: {}, 年龄: {}, 性别 :{}, 课程: {}]'.format(self.name, self.age, self.gender, self.course))

    def set_info(self, x, y, z):
        self.name = x
        self.age = y
        self.gender = z

    def choose(self, course):
        self.course = course


# 类中存放的是对象共有的数据与功能
# 类可以访问
# 1、类的数据属性
# print(Student.stu_school)

# 2、类的函数属性
# print(Student.tell_stu_info)
# print(Student.set_info)


# 但类中的东西是给对象用的
# 1、类的数据属性是共享给所有对象用的，大家访问的地址都一样
# stu_obj = Student('wq', 18, 'male')
# print(stu_obj.name)
# print(stu_obj.age)
# print(stu_obj.gender)
# print(stu_obj.stu_school)

# 类中定义的函数主要是给对象使用的，而且是绑定给对象的，虽然所有对象指向的都是相同的功能，
# 但是绑定到不同的对象就是不同的绑定方法，内存地址各不相同

# 类调用自己的函数属性必须严格按照函数的用法来
# Student.tell_stu_info(stu_obj)

# 绑定方法的特殊之处，谁来调用绑定方法就会将谁当作第一个参数自动传入
# print(Student.tell_stu_info)  # <function Student.tell_stu_info at 0x0228C388>
# print(stu_obj.tell_stu_info())  # <bound method Student.tell_stu_info of <__main__.Student object at 0x0C7543B8>>
# stu_obj.choose('python全栈开发')
# stu_obj.tell_stu_info()