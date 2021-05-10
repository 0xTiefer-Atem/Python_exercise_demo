# p338 random 模块
import random


# print(random.random())  # 大于0小于1之间的小数
#
# print(random.randint(1, 3))  # 1到3闭区间的整数
#
# print(random.randrange(1, 3))  # 左闭右开
#
# print(random.choice([1, '2', [3, 4]]))  # 随机选择
#
# print(random.sample([1, '23', [4, 5]], 2))  # 列表元素两个任意组合
#
# print(random.uniform(1, 3))  # 大于1小于三的整数

# item = [1, 3, 4, 5, 6, 7]
# random.shuffle(item)  # 打乱item的顺序, 相当于 "洗牌"
# print(item)

# 应用: 随机验证码

def make_code(size=4):
    res = ''
    for i in range(size):
        char1 = chr(random.randint(65, 90))
        num1 = str(random.randint(0, 9))
        res += random.choice([char1, num1])
    return res


res = make_code(6)
print(res)
