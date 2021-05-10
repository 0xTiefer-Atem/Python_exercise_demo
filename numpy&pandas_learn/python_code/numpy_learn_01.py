# 数值计算工具NumPy
import numpy as np, time, math  # 导入模块并命名为np
from numpy import array, nan, isnan

# a = np.array([2, 4, 8, 20, 16, 30])  # 单个列表创建一维数组
# # 嵌套元组创建二维数组
# b = np.array(((1, 2, 3, 4, 5), (6, 7, 8, 9, 10),
#               (10, 9, 1, 2, 3), (4, 5, 6, 8, 9.0)))
# print("一维数组：\n", a)
# print("二维数组：\n", b)


# a = np.arange(4, dtype=float)  # 创建浮点型数组：[0., 1.,2., 3.]
# b = np.arange(0, 10, 2, dtype=int)  # 创建整型数组：[0, 2, 4, 6, 8]
# c = np.empty((2, 3), int)  # 创建2×3的整型空矩阵
# d = np.linspace(-1, 2, 5)  # 创建数组：[-1., -0.25,  0.5,  1.25,  2.]
# e = np.random.randint(0, 3, (2, 3))  # 生成[0,3)上的2行3列的随机整数数组
# print("{}\n{}\n{}\n{}\n{}".format(a, b, c, d, e))

# 使用虚数单位j生成数组
# a = np.linspace(0, 2, 5)  # 生成数组：[0.,0.5,1.,1.5,2.]
# b = np.mgrid[0:2:5j]  # 等价于np.linspace(0, 2, 5)
# print("np.linspace: {}\nnp.mgrid: {}".format(a, b))
# print("\n")
# x, y = np.mgrid[0:2:4j, 10:20:5j]  # 生成[0,2]*[10,20]上的4*5的二维数组
# print("x: \n{}\n\ny: \n{}".format(x, y))

# 显示矩阵的各个属性
# a = np.random.randint(1, 11, (3, 5))  # 生成[1,10]区间上3行5列的随机整数数组
# print("矩阵: \n", a)
# print("维数: ", a.ndim)
# print("维度: ", a.shape)
# print("元素总数: ", a.size)
# print("类型: ", a.dtype)
# print("每个元素字节大小: ", a.itemsize)


# 生成数学上一维向量的三种模式
# a = np.array([1, 2, 3])
# print("维度为: {}".format(a.shape))
# b = np.array([[1, 2, 3]])
# print("维度为: {}".format(b.shape))
# c = np.array([[1], [2], [3]])
# print("维度为: {}".format(c.shape))
# 注 形状为(1,n),(n,1),(n,)的array数组意义是不同的；形状为(n,)的一维数组既可以看成行向量，又可以看成列向量，转置不变


# nupmy索引
# (1) 一般索引
# 数组索引示例
# a = np.array([2, 4, 8, 20, 16, 30])
# b = np.array(((1, 2, 3, 4, 5), (6, 7, 8, 9, 10),
#               (10, 9, 1, 2, 3), (4, 5, 6, 8, 9.0)))
# print("a: {}".format(a))
# print(a[[2, 3, 5]])  # 一维数组索引: [ 8 20 30]
# print(a[[-1, -2, -3]])

# print("b: {}".format(b))
# print("\n")
# print(b[1, 2])  # 输出第2行第三列的元素
# print(b[2])  # 输出第三行的元素
# print(b[2, :])  # 输出第三行元素
# print(b[:, 1])  # 输出第二列所有元素
# print(b[[2, 3], 1:4])  # 输出第3，4行，第2，3，4列的元素
# print(b[1:3, 1:3])  # 输出第2，3行，第2，3列的元素

# (2)布尔索引
# a = array([[1, nan, 2], [4, nan, 3]])
# b = a[~isnan(a)]  # 提取a中非nan的数
# print("b= {}".format(b))
# print("b中大于2的元素有: {}".format(b[b > 2]))

# (3)花式索引
# x = array([[1, 2], [3, 4], [5, 6]])
# print("前两行元素为: {}\n".format(x[[0, 1]]))
# print("x[0][0]和x[1][1]为: {}\n".format(x[[0, 1], [0, 1]]))
# print("以下两种格式一样,输出第1，2行，第1，2列元素 ")
# print(x[[0, 1]][:, [0, 1]])
# print("\n")
# print(x[0:2, 0:2])

# 数组修改
# x = np.array([[1, 2], [3, 4], [5, 6]])
# print(x)
# x[2, 0] = -1  # 修改第3行、第1列元素为-1
# print(x)
# y = np.delete(x, 2, axis=0)  # 删除数组的第3行
# print(y)
# z = np.delete(y, 0, axis=1)  # 删除数组的第1列
# print(z)
# t1 = np.append(x, [[7, 8]], axis=0)  # 增加一行
# print(t1)
# t2 = np.append(x, [[9], [10], [11]], axis=1)  # 增加一列
# print(t2)

# 数组降维
# a = np.arange(4).reshape(2, 2)  # 生成数组[[0,1],[2,3]]
# b = np.arange(4).reshape(2, 2)  # 生成数组[[0,1],[2,3]]
# c = np.arange(4).reshape(2, 2)  # 生成数组[[0,1],[2,3]]
# print(a.reshape(-1), '\n', a)  # 输出：[0 1 2 3]和[[0,1],[2,3]]
# print(b.ravel(), '\n', b)  # 输出：[0 1 2 3]和[[0,1],[2,3]]
# print(c.flatten(), '\n', c)  # 输出：[0 1 2 3]和[[0,1],[2,3]]

# 数组组合效果
# a = np.arange(4).reshape(2, 2)  # 生成数组[[0,1],[2,3]]
# b = np.arange(4, 8).reshape(2, 2)  # 生成数组[[4,5],[6,7]]
# c1 = np.vstack([a, b])  # 垂直方向组合
# print(c1)
# print("\n")
# c2 = np.r_[a, b]  # 垂直方向组合
# print(c2)
# print("\n")
# d1 = np.hstack([a, b])  # 水平方向组合
# print(d1)
# print("\n")
# d2 = np.c_[a, b]  # 水平方向组合
# print(d2)
# print("\n")

# 数组分割
# a = np.arange(4).reshape(2, 2)  # 构造2行2列的数组
# b = np.hsplit(a, 2)  # 把a平均分成2个列数组
# c = np.vsplit(a, 2)  # 把a平均分成2个行数组
# print(b, c)

# 数组的简单运算
# a = np.arange(10, 15)
# b = np.arange(5, 10)
# print(a, b)
# c = a + b  # 对应元素相加
# d = a * b  # 对应元素相乘
# print(c, d)
# e = np.modf(a / b)
# print(e)
# print(e[0])  # 取相除的小数位
# print(e[1])  # 取相除的整数位

# 比较运算
# a = np.array([[3, 4, 9], [12, 15, 1]])
# b = np.array([[2, 6, 3], [7, 8, 12]])
# print(a[a > b])  # 取出a大于b的所有元素，输出：[ 3  9  12  15]
# print(a[a > 10])  # 取出a大于10的所有元素，输出：[12  15]
# print(np.where(a > 10, -1, a))  # a中大于10的元素改为-1
# print(np.where(a > 10, -1, 0))  # a中大于10的元素改为-1，否则为0


# ufunc函数效率示例
# x = [i * 0.01 for i in range(1000000)]
# start = time.time()  # 1970纪元后经过的浮点秒数
# for (i, t) in enumerate(x): x[i] = math.sin(t)
# print("math.sin:", time.time() - start)
# y = np.array([i * 0.01 for i in range(1000000)])
# start = time.time()
# y = np.sin(y)
# print("numpy.sin:", time.time() - start)

# 广播机制示例
# a = np.arange(0, 20, 10).reshape(-1, 1)  # 变形为1列的数组，行数自动计算
# print(a)
# b = np.arange(0, 3)
# print(b)
# print(a + b)

# 文本文件存取示例
# a = np.arange(0, 3, 0.5).reshape(2, 3)  # 生成2×3的数组
# np.savetxt("F:\Pdata2_18_1.txt", a)  # 缺省按照'%.18e'格式保存数值，以空格分隔
# b = np.loadtxt("Pdata2_18_1.txt")  # 返回浮点型数组
# print("b=", b)
# np.savetxt("F:\Pdata2_18_2.txt", a, fmt="%d", delimiter=",")  # 保存为整型数据，以逗号分隔
# c = np.loadtxt("Pdata2_18_2.txt", delimiter=",")  # 读入的时候也需要指定逗号分隔
# print("c=", c)

# d = np.loadtxt("Pdata2_19.txt")  # 加载文本数据
# e = d[0:2, 1:4]
# print(e)

# a = np.loadtxt("Pdata2_20.txt", dtype=str, delimiter="，", encoding='utf-8')
# print(a)
# b = a[1:, 1:].astype(float)
# print("b = {}".format(b))

# 读取前6行前8列的值
# a = np.genfromtxt("Pdata2_21.txt", max_rows=6, usecols=range(8))
# b = np.genfromtxt("Pdata2_21.txt", dtype=str, max_rows=6, usecols=[8])  # 读第9列数据
# b = [float(v.rstrip('kg')) for (i, v) in enumerate(b)]  # 删除kg,并转换为浮点型数据
# c = np.genfromtxt("Pdata2_21.txt", skip_header=6)  # 读最后一行数据
# print(a, '\n', b, '\n', c)

# tofile和fromfile存取二进制格式文件示例
# 程序文件Pex2_22.py
# a = np.arange(6).reshape(2, 3)
# a.tofile('Pdata2_22.bin')
# b = np.fromfile('Pdata2_22.bin', dtype=int).reshape(2, 3)
# print(b)

# 存取NumPy专用的二进制格式文件示例
# 程序文件Pex2_23.py
a = np.arange(6).reshape(2, 3)
np.save("Pdata2_23_1.npy", a)
b = np.load("Pdata2_23_1.npy")
print(b)
c = np.arange(6, 12).reshape(2, 3)
d = np.sin(c)
np.savez("Pdata2_23_2.npz", c, d)
e = np.load("Pdata2_23_2.npz")
f1 = e["arr_0"]  # 提取第一个数组的数据
f2 = e["arr_1"]  # 提取第二个数组的数据
print(f1, f2)
