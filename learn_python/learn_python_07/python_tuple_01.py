# 元组：就是"一个不可变的列表"
# 作用：按照索引存放多个值，只用于取不用于改

# 定义：()内用逗号隔开多个任意类型的元素
# t = (1, "q", [1, 2, 3], {"k1": 1, "k2": "2"})  # tuple((1, "q", [1, 2, 3], {"k1": 1, "k2": "2"}))
# print(t)
# print(type(t))  # <class 'tuple'>

# x = (10)  # 单独一个括号表示包含的意思
# print(x, type(x))  # 10 <class 'int'>

# t = (10,)  # 如果元组中只有一个元素，必须加以个逗号

# t = (1, 1.3, "aaa")  # t = (0->值1的内存地址，1->值1.3的内存地址，2->值"aaa"的内存地址)
# t[0] = 111  # 报错: TypeError: 'tuple' object does not support item assignment


# 元组里面存一个列表，只要列表的内存地址不变，列表里的元素可随意改变
# t = (1, 1.3, ["aaa", "bbb"])
# t[2][0] = "cccccc"
# print(t)


# 类型转换：能被for循环的类型都可以被转换为元组
# print(tuple("hello"))
# print(tuple([1,2,3,["aaa", "bbb"]]))
# print(tuple({"k1": 1, "k2": "ccc"}))

# 内置方法
# 按照索引取值(正向+反向)：只能取
t = (1, "q", [1, 2, 3], {"k1": 1, "k2": "2"})
# print(t[1])
# print(t[-2])


# 切片
# print(t[::2])
# print(t[::-1])

# 长度
# print(len(t))


# 成员运算 in ,not in
# print(1 in t)


# 循环
# for item in t:
#     print(item)


# 内置方法 就两个方法
t = (1, 2, 3)
print(t.index(1))
print(t.count(1))

