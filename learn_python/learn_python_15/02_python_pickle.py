# p352
# pickle模块
import pickle

# 序列化
res = pickle.dumps({1, 2, 3, 4, 5, 6})
print(res, type(res))

# 反序列化
re_res = pickle.loads(res)
print(re_res, type(re_res))
