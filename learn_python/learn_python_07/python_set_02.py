# 集合去重
# 局限性
# 1、只能针对不可变类型
# 2、无法保证原来的顺序

# 针对不可变类型，并且保证顺序则需要我们自己写代码实现，例如
# l = [
#     {'name': 'lili', 'age': 18, 'sex': 'male'},
#     {'name': 'jack', 'age': 73, 'sex': 'male'},
#     {'name': 'tom', 'age': 20, 'sex': 'female'},
#     {'name': 'lili', 'age': 18, 'sex': 'male'},
#     {'name': 'lili', 'age': 18, 'sex': 'male'},
# ]
#
# # 列表去重
# new_list = []
# for item in l:
#     if item not in new_list:
#         new_list.append(item)
# print(new_list)

# 集合其他方法
# 1.长度
# s = {'a', 'b', 'c'}
# len(s)

# 2.成员运算
# 'c' in s

# 3.循环
# for item in s:
#     print(item)

# 其他内置方法
s = {1, 2, 3}
# s.discard(4)  # 若删除的元素不存在，不报错
# s.remove(4)  # 若删除的元素不存在，报错
# print(s)


# s.update({1, 3, 5})
# print(s)


s.isdisjoint({4, 5, 6})  # 若两集合交集为空，则返回true
