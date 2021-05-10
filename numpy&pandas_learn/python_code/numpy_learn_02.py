# 文件操作
import numpy as np

# 文件对象属性操作示例。
# 程序文件Pex2_24.py
# f = open("Pdata2_12.txt", "w")
# print("Name of the file:", f.name)
# print("Closed or not:", f.closed)
# print("Opening mode:", f.mode)
# f.close()
n = 0
# 统计文件中的元音字母个数
# with open('Pdata2_25.txt', mode='rt', encoding='utf-8') as f:
#     for line in f:
#         for letter in line:
#             if letter in 'aeiouAEIOU':
#                 n = n + 1
# print("元音个数为: {}".format(n))

# 数据写入文件
# with open('Pdata2_25.txt', mode='wt', encoding='utf-8') as wf:
#     str1 = ['hello', ' ', 'wq']
#     str2 = ['hello', 'aaaa']
#     wf.writelines(str1)
#     wf.write('\n')
#     wf.writelines(str2)
#
# with open('Pdata2_25.txt', mode='rt', encoding='utf-8') as rf:
#     for line in rf:
#         print(line)

a = []
b = []
c = []
with open("../doc/Pdata2_21.txt", mode='rt', encoding='utf-8') as rf:
    for (i, line) in enumerate(rf):
        elements = line.strip().split()
        if i < 6:
            a.append(list(map(float, elements[:8])))
            a.append(float(elements[-1].rstrip('kg')))
        else:
            c = [float(x) for x in elements]
a = np.array(a)
b = np.array(b)
c = np.array(c)
print("{}\n{}\n{}\n".format(a, b, c))
