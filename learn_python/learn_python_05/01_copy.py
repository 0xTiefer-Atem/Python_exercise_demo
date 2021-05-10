# 深浅拷贝
# list1 = [
#     "asdad",
#     "leds",
#     [
#         "1",
#         2
#     ]
# ]

# 这样赋值，二者分隔不开，一个改另一个也跟着改，因为指向的都是同一个东西
# list2 = list1
# list1[0] = "xxxxx"
# print(list2)


# 需求
# 1、拷贝一下原列表产生一个新列表，使两个列表完全独立开(针对改操作独立)

# 如何拷贝出独立列表
# 1、浅拷贝：把原列表第一层内存地址不加区分完全copy一份给新列表
# 对于不可变类型，原列表内容修改，会指向新的内存地址，新列表不会改变
# 但是对于可变类型 比如list，原list改变里面的内容，新列表也会跟着改变
list1 = [
    "asdad",
    "leds",
    [
        "1",
        2
    ]
]

# list2 = list1.copy()
# print(id(list1))
# print(id(list2))

# list1[0] = "ASDAD"
# list1[1] = "LEDS"
# list1[2] = "xxxxx"
# list1[2][0] = "111"
# list1[2][1] = 222

# print(list1)
# print(list2)


# 深拷贝
# 对于不可变类型拷贝原来的地址
# 对于可变类型，容器生成新的，里面的数据拷贝原容器里的数据地址
import copy

list1 = [
    "asdad",
    "leds",
    [
        "1",
        2
    ]
]

list3 = copy.deepcopy(list1)
# print(id(list1))
# print(id(list3))

print(id(list1[0]), id(list1[1]), id(list1[2]))
print(id(list3[0]), id(list3[1]), id(list3[2]))

