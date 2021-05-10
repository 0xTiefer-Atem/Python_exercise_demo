# 控制文件指针操作
# 指针移动的单位是字节
# 只有一种模式特殊:
#               t模式下的read(n),n代表字符个数

# with open('../doc/test.txt', mode='rt', encoding='utf-8') as f1:
#     res = f1.read(3)
#     print(res)

# f.tell() 获取文件指针当前位置

# f.seek(n,model)
# n是指指针移动的个数
# model:
#       0: 文件指针从文件头开始
#       1: 文件指针从当前位置开始
#       2: 文件指针从文件末尾开始，应该让指针倒着移动
# 强调: 只有0模式可以在t模式下使用,1、2只能在b模式下使用

# with open(r'../doc/test.txt', mode='rb') as f2:
#     print(f2.tell())
#     f2.seek(100, 0)
#     res = f2.read()
#     print(res.decode('utf-8'))

# 案例
# 模拟linux里的 tail -f 命令
