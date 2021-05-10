# 字符串类型的操作
# 定义
# msg = 'hello'
# msg = str('hello')
# 类型转换
# str()函数可以把任意其他类型转化为字符串类型
# dic_str = str({
#     "k1": 2
# })
# print(dic_str)
# 内置方法
# 1、按索引取值(正向取+反向取) 只能取值不能修改
# msg = "hello world"

# 正向取
# print(msg[0])

# 反向取
# print(msg[-1])


# 2、切片(顾头不顾尾,步长) 索引的拓展应用
# msg = "hello world"
# res = msg[0:5]  # 顾头不顾尾
# print(res)


# res = msg[0:5:2]  # 最后一个参数是步长
# print(res)


# 反向步长
# res = msg[5:0:-1]
# print(res)


# 效果相同
# res = msg[0:11]
# res = msg[:]


# 反转字符串
# res = msg[::-1]
# print(res)


# 3、字符串长度
# msg = "hello world"
# print(len(msg))


# 4、成员运算 in 和 not in
# 判断一个子字符串是否在一个大字符串中
# print("wq" in "wq is nb")
# print("wq" not in "wq is nb")


# 5、移除字符串两端指定符号，但原来的字符串不会改变
# msg = "s=+/()       adsasd asdasdasd           =+/()s"
# res = msg.strip('s=+/() ')  # 函数默认去掉空格，可以传参数
# print(res)


# 6、切分 split() 把字符串按照某种分隔符进行切分 默认切分空格，返回list
# msg = "hello world asdasda"
# res = msg.split(" ")
# print(res)

# 指定分割次数
# res = msg.split(" ", 1)
# print(res)


# 需要掌握的操作
# 1、strip、lstrip、rstrip
# msg = "***wq***"
# print(msg.strip("*"))  # 去掉两侧的制定字符
# print(msg.lstrip("*"))  # 只去掉左侧指定字符
# print(msg.rstrip("*"))  # 只去掉右侧指定字符

# 2、lower，upper
# msg = "Wwq"
# print(msg.upper())  # 转化为大写
# print(msg.lower())  # 转化为小写


# 3、startswith()、endswith()
# msg = "hello wq world hahaha"
# print(msg.startswith("hello"))  # 是否以特定字符开头
# print(msg.endswith("ha"))  # 是否以特定字符结尾

# 4、split()、rsplit()
# msg = "wq:18:m"
# print(msg.split(":", 1))  # ['wq', '18:m']
# print(msg.rsplit(":", 1))  # ['wq:18', 'm']


# 5、join把列表拼接成字符串
# l1 = ['wq', '18:m']
# res = ":".join(l1)  # 按照某个分隔符，把列表里的全部元素拼接成一个大字符串
# print(res)


# 6、repalce
# msg = "you can you up no can no bb"
# res = msg.replace("you", "YOU", 1)  # 最后一个参数是替换几次，默认全部替换
# print(res)


# 7、isdigit
# 判断字符串是否有纯数字组成
# msg = "123.56"
# print(msg.isdigit())


# 8、了解
# find、rfind、index、rindex、count
msg = "hello wq hahahahaha"

# 找到返回起始索引
# print(msg.find("l"))  # 返回子字符串在原字符串左侧第一次出现的位置
# print(msg.find("wq"))
#
#
# print(msg.index("l"))
# print(msg.index("wq"))


# 找不到情况
# print(msg.find("x"))  # 找不到返回-1

# print(msg.index("x"))  # 找不到直接抛出异常


# print(msg.count("ha"))  # 统计字符出现次数

# center,ljust,rjust,zfill
# 显示效果
# print("wq".center(10, "-"))  # 效果: ----wq----
# print("wq".ljust(10, "-"))  # 效果: wq--------
# print("wq".rjust(10, "-"))  # 效果: --------wq
# print("wq".zfill(10))  # 效果: 00000000wq


# expandtabs
# 制定制表符宽度
# msg = "hello\tworld"
# print(msg)                 # 效果： hello	world
# print(msg.expandtabs(2))   # 效果： hello world
# print(msg.expandtabs(10))  # 效果： hello     world


# captalize 首字母大写
# msg = "hello wq"
# print(msg.capitalize())  # 效果: Hello wq


# swapcase 大小写反转
# msg = "HEllo wQ"
# print(msg.swapcase())  # 效果: heLLO Wq


# title 每个单词首字母大写，以空格为分隔符
# msg = "hello wq"
# print(msg.title())  # 效果: Hello Wq


# 字符串里的is相关函数
# print("abc".islower())  # 判断字符串是否全为小写
# print("abc".isupper())  # 判度字符串是否全为大写
# print("abc".istitle())  # 判断字符串里的每个单词是否为大写
# print("abc".isalnum())  # 判断字符串是否只由字母或数字构成
# print("abc".isalpha())  # 判断字符串是否只由字母组成
# print("abc".isspace())  # 判断字符串是否只为空格
# print("age_wq".isidentifier())  # 判断字符串命名是否规范，包括python自带的关键字
# print("def".isidentifier())  # 判断字符串命名是否规范，包括python自带的关键字


# 字符串的与数字有关的is函数
num1 = b'4'  # bytes
num2 = u'4'  # unicode,python3中无需加u就是unicode
num3 = '四'  # 中文数字
num4 = 'Ⅳ'  # 罗马数字

# isdigt可识别：bytes,unicode
print(num1.isdigit())  # True
print(num2.isdigit())  # True
print(num3.isdigit())  # False
print(num4.isdigit())  # False

# isdecimal只能识别: uncicode(bytes类型无isdecimal方法)
# num2.isdecimal()  # True
# num3.isdecimal()  # True
# num4.isdecimal()  # True

# isnumberic:可识别 unicode,中文数字,罗马数字(bytes类型无isnumberic方法)
# num2.isnumeric()  # True
# num3.isnumeric()  # True
# num4.isnumeric()  # True

# 三者不能判断浮点数
# num5 = '4.3'
# num5.isdigit()
# num5.isdecimal()
# num5.isnumeric()
