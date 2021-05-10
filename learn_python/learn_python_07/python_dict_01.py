# 字典类型
# 定义：{}内用逗号隔开多个key，value，其中value是任意类型，但key必须是不可变类型，且不能重复
# dis = {"k1": 1, "k2": [1, 2, 4], "k3": {"v1": 19}}

# d = {}  # 默认定义出来的是空字典

# d = dict()
# print(d, type(d))
# d = dict(x=1, y=2, z="qwe")
# print(d)


# 数据类型转换
# info = [
#     ["name", "wq"],
#     ("age", 18),
#     ["gender", "male"]
# ]

# d = {}

# 方式一
# for item in info:
#     d[item[0]] = item[1]
# print(d)

# 方式二  利用解压操作
# for k, v in info:
#     d[k] = v
# print(d)


# res = dict(info)
# print(res)


# 创造一个value为None的字典
# 方式一
# keys = ["name", "age", "gender"]
# values = None
# d = {}
# for item in keys:
#     d[item] = values
# print(d)


# 方式二
# print({}.fromkeys(keys))


# 内置方法
# 按key取值：可存可取
# d = {"k1": 111}
# # 若key存在则修改
# d["k1"] = 2222
# # 若不存在则添加
# d["k2"] = 333
# print(d)
#
#
# # 长度
# print(len(d))
#
#
# # 成员运算 in， not in
# print("k1" in d)
# print(222 in d)


# 删除
# d = {"k1": 111, "k2": 2222, "k3": "Qweqwe"}

# 通用删除
# del d["k1"]
# print(d)

# 字典自带的删除方法
# pop函数根据key删除元素,并返回删除key对应的值
# res = d.pop("k1")
# print(d)
# print(res)


# popitem随机删除，并返回随机删除的key，value所组成的元组
# res = d.popitem()
# print(res, type(res))  # ('k3', 'Qweqwe') <class 'tuple'>

# d = {"k1": 111, "k2": 2222, "k3": "Qweqwe"}

# keys(), values(), items() 方法
# for key in d.keys():
#     print(key)

# for value in d.values():
#     print(value)

# for x, y in d.items():
#     print(x, y)


# 其他方法
d = {"k1": 111, "k2": 2222, "k3": "Qweqwe"}
# d.clear()
# print(d)

# 直接取值d["k1"]若key不存在则直接报错，get方法不会
# res = d.get("k4")
# print(res)


# 若d里面没有k1则添加k1并设置默认value 1111，若有就不进行操作   函数执行完之后返回k1所对应的值
d.setdefault("k1", "asdasdasd")



# dic = {"asd": 222, "k1": "sdfsdf"}
# d.update(dic)
# print(d)
