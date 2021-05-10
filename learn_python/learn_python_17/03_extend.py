# p373 继承的实现

class Student:
    school = '辽宁石油化工大学'

    def __init__(self, name, age, gender):
        self.name = name
        self.age = age
        self.gender = gender


    def choose_course(self):
        print(r'{} 正在选课'.format(self.name))



class Teacher:
    school = '辽宁石油化工大学'

    def __init__(self, name, age, gender, salary, level):
        self.name = name
        self.age = age
        self.gender = gender
        self.salary = salary
        self.level = level

    def score(self):
        print(r'{} 正在给学生打分'.format(self.name))