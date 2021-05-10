# 文件操作模式详解
# 以t模式为基准进行文件操作
# 1、r(默认操作模式)，只读模式,当文件不存在时,报错,当文件存在时，文件指针跳到最开始位置
# with open('../doc/test.txt', mode='rt', encoding='utf-8') as f:
#     print('第一次读'.center(50, "*"))
#     res = f.read()  # 不加参数，一次性将数据读入内存
#     print(res)
#     print('第二次读'.center(50, "*"))
#     res1 = f.read()
#     print(res1)

# 循环读取文件的每一行
# with open("../doc/test2.txt", mode='rt', encoding='utf-8') as file2:
#     for line in file2:
#         username, password = line.strip().split(':')
#         print('username: {}, passw'
#               'ord: {}'.format(username, password))

# 从文件读取用户名信息验证
# input_username = input('input your username: ')
# input_password = input('input your password: ')
# with open("../doc/test2.txt", mode='rt', encoding='utf-8') as file1:
#     username, password = file1.read().split(":")
# if username == input_username and password == input_password:
#     print("login success")
# else:
#     print('login error')


# 2、w: 只写模式，当文件存在时会在路径下创建空文件，当文件存在时,w模式会清空文件内容,指针位于开始位置
# with open('../doc/xx.txt', mode='wt', encoding='utf-8') as file:
#     # file.read()  # io.UnsupportedOperation: not readable
#     file.write("asdasdadad")

# 注意1: 在以w打开文件，在没有关闭的情况下，连续写,新写入的内容总是在旧的之后
# with open('../doc/xx.txt', mode='wt', encoding='utf-8') as file:
#     # file.read()  # io.UnsupportedOperation: not readable
#     file.write("asdasdadad\n")
#     file.write("asdfasdfas\n")
#     file.write("123434523\n")


# 3、a: 只追加写，在文件不存在时,会创建空文档，在文件存在时，文件指针会直接跳到内容末尾
# with open('../doc/x.txt', mode='at', encoding='utf-8') as f:
#     # f.read()  # 报错，不能读
#     f.write("哈哈哈\n")

# 强调 w 模式与 a 模式的异同：
# 1 相同点：在打开的文件不关闭的情况下，连续的写入，新写的内容总会跟在前写的内容之后
# 2 不同点：以 a 模式重新打开文件，不会清空原文件内容，会将文件指针直接移动到文件末尾，新写的内容永远写在最后

# 小案例: a模式
# 注册功能
# username = input("请输入用户名: ")
# password = input("请输入密码: ")
# with open('../doc/account.txt.txt', mode='at', encoding='utf-8') as f:
#     f.write('{}:{}\n'.format(username, password))


# 案例: w模式用来创建全新的文件
# 文本文件的copy工具
# src_file_path = input("源文件路径>>: ").strip()
# target_file_path = input("目标文件路径>>: ").strip()
# with open(r'{}'.format(src_file_path), mode='r', encoding='utf-8') as rf, \
#         open(r'{}'.format(target_file_path), mode='w', encoding='utf-8') as wf:
#     res = rf.read()
#     wf.write(res)


# 了解: +不能单独使用,必须配合r,w,a
