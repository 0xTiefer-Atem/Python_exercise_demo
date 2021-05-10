# x模式: 只写模式、不可读，不存在则创建，存在就报错

#  文件存在情况下 FileExistsError: [Errno 17] File exists: '../doc/x.txt' 直接报错
# with open('../doc/x.txt', mode='x', encoding='utf-8') as x_file:
#     pass

# 不存在就创建
# with open('../doc/b.txt', mode='x', encoding='utf-8') as x_file:
#     ...

"""
t:
  1、读写都是以字符串形式的
  2、只能针对文本文件
  3、必须指定字符编码
b:binary模式
  1、读写都是以bytes为单位
  2、可以针对所有文件
  3、一定不能指定encoding参数
"""
# 读模式
# with open(r'../doc/x.txt', mode='rb') as f1:
#     res = f1.read()  # utf-8的二进制
#     print(res)
#     print(type(res))  # bytes类型  <class 'bytes'>
#     print(res.decode('utf-8'))

# 写模式
# with open(r'../doc/test1.txt', mode='wb') as f2:
#     f2.write('王骞'.encode('utf-8'))

# 二进制模式文件拷贝工具
src_file_path = input("请输入源文件路径: ")
target_file_path = input("请输入目标文件路径: ")
with open(r'{}'.format(src_file_path), mode='br') as src_file, \
        open(r'{}'.format(target_file_path), mode='bw') as target_file:
    for line in src_file:
        target_file.write(line)

# 补充: 循环读取文件
# while 方式: 可控制每次读入的大小
# with open(r'../doc/test.jpg', mode='br') as img_file:
#     while True:
#         res = img_file.read(1024)
#         if len(res) == 0:
#             break
#         print(len(res))

# for 方式
# with open(r'../doc/test.jpg', mode='br') as img_file:
#     for line in img_file:
#         print(len(line))


# 其他文件操作方法
# 读相关操作补充
# f.readline()  # 读取一行内容,光标移动到第二行首部
# f.readlines()  # 读取每一行内容,存放于列表中

# 写相关操作补充
# f.writelines(['333\n','444\n'])  # 文件模式
# f.writelines([bytes('333\n',encoding='utf-8'),'444\n'.encode('utf-8')]) #b模式
