# p348 序列化与反序列化的介绍
# 1、序列化
# 序列化指的是把内存中的数据类型转化成一种特定的格式的内容
# 该格式的内容可用于存储或者传输给其他平台的使用
# 序列化: 内存中的数据模型 ----> 序列化 ----> 特定格式 (json格式或者pickle格式)
# 反序列化: 特定格式 (json格式或者pickle格式) ----> 序列化 ----> 内存中的数据模型

# 2、序列化用途
#   1) 可用于存储
#   2) 传输给其他平台使用

# p349
# 3、json模块的基本使用
import json

# 序列化
# json_res = json.dumps([1, 'aaa', True, False, {"k1": 2, "k2": 'qwe'}])
# print(json_res, type(json_res))  # [1, "aaa", true, false, {"k1": 2, "k2": "qwe"}] <class 'str'>
# with open(r'./doc/test.txt', mode='wt', encoding='utf-8') as json_f:
#     json_f.write(json_res)

# 序列化的简单方法
# with open(r'./doc/test.txt', mode='wt', encoding='utf-8') as json_f:
#     json.dump([1, 'aaa', True, False, {"k1": 2, "k2": 'qwe'}], json_f)


# 反序列化
# with open('./doc/test.txt', mode='rt', encoding='utf-8') as re_json_f:
#     res = re_json_f.read()
#     re_json_res = json.loads('[1, "aaa", true, false, {"k1": 2, "k2": "qwe"}]')
#     print(re_json_res, type(re_json_res))


# 反序列化的简单方法
# with open('./doc/test.txt', mode='rt', encoding='utf-8') as re_json_f:
#     re_json_res = json.load(re_json_f)
#     print(re_json_res, type(re_json_res))


# p350
# json格式的补充
# json合适兼容的是所有语言通用的数据类型, 不能识别单一语言的所独有的类型
# json.dumps({1, 2, 3, 4, 5})  # TypeError: Object of type set is not JSON serializable

# json强调

# p351
# 猴子补丁
