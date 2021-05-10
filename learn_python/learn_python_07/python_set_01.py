# 集合
# 作用：去重，关系运算
# 关系运算
# 给出两个人的关系圈，找出共同好友
# friends1 = {"zero", "kevin", "jason", "egon"}  # 用户1的好友们
# friends2 = {"Jy", "ricky", "jason", "egon"}  # 用户2的好友们
# l = []
# for x in friends1:
#     if x in friends2:
#         l.append(x)
# print(l)

# 1.2去重


# 定义：在{}内用逗号分隔开多个元素。，集和具备以下三个特点:
# 1: 每个元素必须是不可变类型
# 2: 集合内没有重复元素
# 3: 集合内元素无序

# 注意1: 列表类型是索引对应值，字典是key对应值，均可以取得单个指定的值，
#       而集合类型既没有索引结构也没有key对应的值，所以无法取得单个值，而且对于
#       集合来说，主要是用于去重，根本没有取出单个值的需求

# 注意2: {}既可以用于定义字典，也可用于定义集合，但是字典里必须是kv键值对，
# d = {}  # 默认是空字典
# s = set()  # 空集合

# 使用
# s = {1, 2, 3, 4}  # 等价于 s = set({1,2,3,4})
# print(s, type(s))  # {1, 2, 3, 4} <class 'set'>

# 类型转换
# res = set("helloqwedfrt")
# print(res)

# 但凡能被for循环的遍历的数据类型（强调：遍历出的每一个值都必须为不可变类型）都可以传给set()转换成集合类型
# s = set([1, 2, 3, 4])
# s = set((1, 2, 3, 4))
# print(s)

# 内置方法
friends1 = {"zero", "kevin", "jason", "egon"}  # 用户1的好友们
friends2 = {"Jy", "ricky", "jason", "egon"}  # 用户2的好友们

# 1：取交集：两者共同好友
# res = friends1 & friends2
# res = friends1.intersection(friends2)
# print(res)

# 2：取并集，两个所有好友
# res = friends1 | friends2
# res = friends1.union(friends2)
# print(res)

# 3：差集(有左右顺序之分)，去friends1独有的好友
# res = friends1 - friends2
# res = friends1.difference(friends2)
# print(res)

# 4：对称差集(^) # 求两个用户独有的好友们（即去掉共有的好友）
# res = friends1 ^ friends2
# res = friends1.symmetric_difference(friends2)
# print(res)

# 5：父子集：包含关系
# s1 = {1, 2, 3}
# s2 = {1, 2, 4}
# 不存在包含关系，下面比较均为false
# print(s1 > s2)
# print(s1 < s2)

# s1 = {1, 2, 3}
# s2 = {1, 2}
# print(s1.issuperset(s2))  #父级
# print(s1.issubset(s2))  # 子集
# print(s1 > s2)  # s1是s2的父


# s1 = {1, 2, 3}
# s2 = {1, 2, 3}
# print(s1 == s2)  # s1,s2互为父子
