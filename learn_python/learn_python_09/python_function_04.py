# 作业
# 1、写函数，统计传入的字符串中 数字、字母、空格、以及 其他的个数
# def str_count(my_str):
#     dis = {
#         'number': 0,
#         'letter': 0,
#         'blank': 0,
#         'other': 0,
#     }
#     for item in my_str:
#         if item.isspace():
#             dis['blank'] = dis['blank'] + 1
#         elif item.isalpha():
#             dis['letter'] = dis['letter'] + 1
#         elif item.isdigit():
#             dis['number'] = dis['number'] + 1
#         else:
#             dis['other'] = dis['other'] + 1
#     print(dis)
#
#
# str_count('1q* ')


# 2、判断传入的参数 (字符串、列表、元组) 长度是否大于5
# def data_len(args):
#     return len(args) > 5
#
#
# print(data_len((1,2,2,3,3,3,3,3)))


# 3、检查传入列表长度，若大于2，就返回前两个长度内容
# def new_data(obj):
#     if len(obj) >= 2:
#         l = []
#         l.append(obj[0])
#         l.append(obj[1])
#         return l
#
#
# print(new_data('asdasdasd'))


# 4、返回传入参数的奇数索引数据
# def odd_index_element(obj):
#     l = []
#     for item in range(len(obj)):
#         if item % 2 != 0:
#             l.append(obj[item])
#     return l
#
#
# print(odd_index_element(['a', 'd', 'e', 'a', '9']))


# 5、返回字典中value长度大于2的键值对
# def new_dic(obj):
#     dit = {}
#     for key in obj:
#         if len(obj[key]) > 2:
#             dit[key] = obj[key][0:2]
#     return dit
#
#
# print(new_dic({'a': 'v1v2', 'b': '12345', 'c': '8'}))
