# 索引选取
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

# df = pd.read_csv('../doc/NBAPlayers.txt', sep='\t')
# print(df[:5])

# loc行选取
# print(df.loc[1])

# iloc 基于索引
# print(df.iloc[5])

# loc对列操作  逗号后面是对列进行操作
# print(df.loc[:3, ["Player", 'height']])

# iloc对列操作  基于下标序号
# print(df.iloc[:3, :3])


# loc数据过滤
# 根据条件选取数据
# print(df.loc[(df["height"] >= 180) & (df['weight'] >= 80)])

# 分组计算,统计函数
df = pd.read_csv('../doc/movie.csv')
grouped = df.groupby("director_name")
print(grouped.size())

# %matplotlib inline

x = [13854, 12213, 11009, 10655, 9503]  # 程序员工资，顺序为北京，上海，杭州，深圳，广州
x = np.reshape(x, newshape=(5, 1)) / 10000.0
y = [21332, 20162, 19138, 18621, 18016]  # 算法工程师，顺序和上面一致
y = np.reshape(y, newshape=(5, 1)) / 10000.0
# 调用模型
lr = LinearRegression()
# 训练模型
lr.fit(x, y)
# 计算R平方
print( lr.score(x,y))
# 计算y_hat
y_hat = lr.predict(x)
# 打印出图
plt.scatter(x, y)
plt.plot(x, y_hat)
plt.show()
