# 1、九九乘法表

# for i in range(1, 10):
#     for j in range(1, i+1):
#         print("{i} * {j} = {res}".format(i=i, j=j, res=i * j), end="      ")
#     print()


# 2、打印金字塔
# 数学表达式
# 空格数 = max_level - current_level
# *号数 = 2 * current_level - 1
# max_level = 5
# for current_level in range(max_level):
#     for k in range(max_level - current_level):
#         print(" ", end="")
#     for j in range(2 * current_level - 1):
#         print("*", end="")
#     print()


# 3、字符串操作
name = " aleX"
# 1) 移除name变量对应的值两边空格
# res = name.strip()
# print(res)
# 2) 判断name是否以al开头
# res = name.startswith("al")
# print(res)
# 3) 判断name是否以X结尾
# res = name.endswith("X")
# print(res)
# 4) 将name中的l替换为p
# res = name.replace("l", "p")
# print(res)
# 5) 将name以l进行分割
# res = name.split("l")
# print(res)
# 6) 将name变为大写
# res = name.upper()
# print(res)
# 7) 将name变为小写
# res = name.lower()
# print(res)
# 8) 输出name第二个字符
# res = name[2]
# print(res)
# 9) 输出name前三个字符
# res = name[0:3:1]
# print(res)
# 10) 输出name后两个字符
# res = name[4:2:-1]
# res = name[-2:]
# print(res)
# 11) 输出name中e所在的索引值
# res = name.index("e")
# print(res)
# 12) 去掉最后一个字符
res = name[0:4]
res = name[:-1]
print(res)
