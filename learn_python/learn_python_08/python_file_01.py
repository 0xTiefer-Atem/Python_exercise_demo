# 文件
# 文件操作基本流程
# 1、打开文件
# 若不指定绝对路径,则会从当前py文件的路径下开始找
# 文件的相对位置，打开文件的模式，打开文件的编码
file = open('../doc/test.txt', mode='rt', encoding="utf-8")
print(file)
# 2、操作文件: 读或者写
res = file.read()
print(res)
# 3、关闭文件
file.close()
# del file  # 可不需要
