# 修改文件的两种方式
# with open('../doc/a.txt', mode='r+t', encoding='utf-8') as f1:
#     f1.seek(9,0)
#     f1.write('<主任>')

# 方式一: 文本编辑器采用的就是这种方式
# with open('../doc/test1.txt', mode='rt', encoding='utf-8') as f1:
#     res = f1.read()
#     data = res.replace('x', 'aaa')
#     print(data)
#
# with open('../doc/test1.txt', mode='wt', encoding='utf-8') as f2:
#     f2.write(data)


# 方式二
import os
with open('../doc/test1.txt', mode='rt', encoding='utf-8') as f2, \
        open('../doc/.test1.txt.swap', mode='wt', encoding='utf-8') as f3:
    for line in f2:
        res = line.replace('x', 'aaa')
        f3.write(res)
os.remove('../doc/test1.txt')
os.rename('../doc/.test1.txt.swap', '../doc/test1.txt')