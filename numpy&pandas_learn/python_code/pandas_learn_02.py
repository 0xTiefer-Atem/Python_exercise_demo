# filter函数
# 对下列数据只保留能被2整除的数
# x = [1, 3, 5, 7, 2, 19, 8, 18, 45, 24]
# print(x)
# x = list(filter(lambda s: s % 2 == 0 or s % 3 == 0, x))
# print(x)

import pandas as pd
import numpy as np
import random

# df = pd.read_csv('../doc/NBAPlayers.txt', sep="\t")
# print(df.head(2))
l = []
for i in range(0, 325):
    l.append(random.uniform(0, 2))
print(l)
df = pd.DataFrame(l,columns=['RON损失值'], index=None)
df.to_excel('数据.xlsx')

# 缺省值的计算  Missing Value
# print(pd.isnull(df))  # 对所有列进行判断
# print(pd.isnull(df["Player"]))  # 对指定列判断

# 删除缺失值的行
# df.dropna()


# 文本数据处理
# s = pd.Series(['a', 'B', 'asd', ' Baca', 'dog', 'cat'])
# print(s.str.strip())
# print(s.str.upper())
# print(s.str.strip())
# print(s[s.str.strip().str.endswith('a')])
# a = {"name": ['xiaoming', 'xiaohong', 'xiaobai'], ' age': [13, 14, 15]}
# df1 = pd.DataFrame(data=a)
# df1.columns = df1.columns.str.strip()  # 去掉列两端空格
# print(df1['age'])

# 字符串分割
# print(df["Player"].str.split(" ").get(0))
