# p356 hash介绍
"""
hash
    hash是一类算法, 该算法接受传入的内容, 经过运算的到一串hash值
特点:
    1、只要传给的内容一样, 则结果就一定一样
    2、不能由hash值反解回内容
    3、只要使用的hash算法不变, 无论传入的内容有多大, 得到的hash长度不变

用途:
    1、密码加密
    2、文件完整性校验
"""

# p357 hashlib使用
import hashlib

m = hashlib.md5()

# 放入需要hash的原材料
m.update("hello".encode('utf-8'))

# 获取hash之后的值
res = m.hexdigest()
# print(res)

# p358 密码加盐

# 模拟撞库
# 抓包获取的加密密码
cryptograph = 'aee949757a2e698417463d47acac93df'

# 猜测的密码列表
pwds = [
    'alex3714',
    'alex1313',
    'alex94139413',
    'alex123456',
    '123456alex',
    'a123lex',
]
dic = {}
# for pwd in pwds:
#     res = hashlib.md5(pwd.encode('utf-8'))
#     dic[pwd] = res.hexdigest()
# print(dic)
#
# for k, v in dic.items():
#     if v == cryptograph:
#         print("撞库成功, 明文密码为: {}".format(k))


# 提高安全性---->密码加盐


# p359 subprocess模块
# 执行系统命令
import subprocess
obj = subprocess.Popen('ls', shell=True, stdout=subprocess.PIPE,stderr=subprocess.PIPE)
# res = obj.stdout.read()
# print(res.decode('gbk'))

err_res = obj.stderr.read()
print(err_res.decode('gbk'))
