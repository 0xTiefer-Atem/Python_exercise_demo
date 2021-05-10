# 列表类型
# 定义
# l = [1, 2.3, "a", [1, 2, 3, 4], {"k1": 1}]  # l = list([1, 2.3, "a", [1, 2, 3, 4], {"k1": 1}]) 作用相同
# print(type(l))

# 类型转换：但凡能够被for循环遍历的类型，都可以当作参数传给list()转成列表
# res = list("hello")
# res = list()
# res = list({"k1": 1, "k2": 2})
# print(res)

# 函数
# 1、按照索引取值(正向+反向)：可取可改
# l = [1, 2.3, "a", [1, 2, 3, 4], {"k1": 1}]

# 正向取值
# print(l[0])

# 反向取值
# print(l[-1])

# 可存可改
# l[0] = 222
# print(l)

# 2、切片(顾头不顾尾，步长) 切片就是拷贝行为，相当于浅拷贝
# l = [1, 2.3, "a", [1, 2, 3, 4], {"k1": 1}]
# res = l[0:2]
# res = l[0:2]
# print(res)

# print(l[::-1])  # 倒置整个列表


# 3、长度
# print(len(l))

# 4、成员运算 in、not in
# print("a" in l)
# print("c" not in l)

# 5、追加 永远在末尾添加
# l.append("asdasdasd")
# print(l)

# 6、插入值
# l.insert(2, "aaaaaaaaa")
# print(l)

# 7、往list里添加list
# new_l = ["1", "asd", "4rt"]
# 循环实现
# for item in new_l:
#     l.append(item)
# print(l)

# 函数实现
# l.extend(new_l)
# print(l)


# 、删除
# l = [1, 2.3, "a", [1, 2, 3, 4], {"k1": 1}]

# 方式1 通用删除方法，没有返回值
# del l[1]
# print(l)

# 方式2 l.pop() 根据索引删除，不指定索引，默认删除最后一个
# item = l.pop()
# item1 = l.pop(2)
# print(item1, l)

# 方式3 l.remove() 根据指定元素删除，没有返回值
# l.remove("a")
# print(l)


# 循环
# for item in l:
#     print(item)


# 需要掌握的操作
# l = [1, 2.3, "a", [1, 2, 3, 4], {"k1": 1}]

# l.count()
# 统计元素出现了几次，仅限于第一层元素里
# print(l.count(1))


# l.index()
# 查找某一元素索引,找不到报错
# print(l.index(1))


# l.clear()
# 清空列表里的元素
# l.clear()
# print(l)


# l.reverse() 不是排序就是将列表倒置
# 反转列表
# l.reverse()
# print(l)


# l.sort()
# 对列表元素进行排序, 默认升序，列表内元素必须是同种类型才能排序
# l = [-11, -9, 10, -12, 1, 2, 4]
# l.sort()
# l.sort(reverse=True)  # 设置为降序
# print(l)

# 字符串可以比大小，根据ASCII

# 列表之间比大小的原理同字符串，但必须对应位置也是同种类型


# 用列表模拟堆栈
# # 队列
# l = []
#
# # 入队操作
# l.append("1")
# l.append("2")
# l.append("3")
#
# print(l)
#
# # 出队操作
# l.pop(0)
# print(l)



# 栈
l = []

# 入队操作
l.insert(0, "1")
l.insert(0, "2")
l.insert(0, "3")

print(l)

# 出队操作
l.pop(0)
print(l)
