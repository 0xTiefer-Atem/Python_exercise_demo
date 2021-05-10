import pandas as pd
import numpy as np

# print(pd.__version__)
# 一些注意点
# 不要城市读取excel文件,最好使用通用的csv或者txt文件格式
# 注意编码问题,使用encoding参数
# 注意处理报错行
#
# df = pd.read_csv('../doc/NBAPlayers.txt', sep="\t")
# print(df.head(3))
# print(type(df))
# print(df.Player)

# Series的创建
# 例一 将字典转换为Series
# a = {'name': 'wq', 'age': 18, 'height': '180'}
# pd_series = pd.Series(a)
# print(pd_series)
# b = [1, 2, 3, 4, 5, 6]
# pd_series = pd.Series(b, index=list("abcdef"))  # 指定索引下标
# print(pd_series)

# DataFrame的创建
# a = {"name": ['xiaoming', 'xiaohong', 'xiaobai'], 'age': [13, 14, 15]}
# df1 = pd.DataFrame(data=a, index=list('ABC'))
# print(df1)

# b = [[1, 2, 3, 4, 5, 6], ['a', 'b', 'c', 'd', 'e', 'f']]
# df2 = pd.DataFrame(data=b, index=list('AB'), columns=list('ABCDEF'))
# print(df2)


# DataFrame常用操作1
# df = pd.read_csv('../doc/NBAPlayers.txt', sep="\t")
# 常见的属性
# print(df.columns)
# print(df.index)
# print(df.dtypes)  # 返回每列的数据类型
# print(df.shape)  # 返回行和列
# print(df.size)  # 返回总数据
# print(len(df))  # 返回行数

# 常用的方法
# print(df.head())  # 默认返回前五行
# print(df.tail())  # 默认返回后五行
# df = df.rename(columns={"height": "Height"})
# print(df)

# print(df.replace({"Player": {"Curly Armstrong": "wq"}}))  # 字典套字典: {列名:{旧值:新值}}
# print(df.collage.value_counts())  # 对应该列每个元素出现的次数
# print(df.sort_values(by=["height", "weight", "collage"]))
# print(df.describe())


# DataFrame常用操作2
# df = pd.read_csv('../doc/NBAPlayers.txt', sep="\t")
# print(df.head(2))
# 选择数据
# print(df["Player"])
# print(df[["Player", "height"]])

# 增加一列
# df["class"] = 1
# print(df.head(2))

# 条件筛选
# print((df["height"] >= 200) | (df["height"] <= 170))
# print(df[(df["height"] >= 200) | (df["height"] <= 170)])

# 行列求和
# 属性 axis  0，行  1，列
# 默认是0
# print(df.sum(axis=0))  # 对这列求和
# print(df.sum(axis=1))  # 对这行求和


# a = np.array([[1, 2, 3, 4, 5], [67, 8, 9, 0, 19], [11, 12, 13, 141, 5]])
# print(np.sum(a, axis=0))  # 列求和
# print(np.sum(a, axis=1))  # 行求和
# print(np.sum(a,))  # 全部求和
