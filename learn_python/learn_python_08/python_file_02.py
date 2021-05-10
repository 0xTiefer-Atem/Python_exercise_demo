# with的上下文管理
# 自动执行file.close()
# 用法1
# with open("../doc/test.txt", mode='rt', encoding='utf-8') as file1:
#     res = file1.read()
#     print(res)


# 用法2 with打开多文件
res1 = ''
with open("../doc/test.txt", mode='rt', encoding='utf-8') as file1, \
        open("../doc/test2.txt", mode='rt', encoding='utf-8') as file2:
    res = file1.read()
    res1 = file2.read()

print(res1)
