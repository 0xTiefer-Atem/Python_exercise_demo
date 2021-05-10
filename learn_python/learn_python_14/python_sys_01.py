# p342 sys模块
# import sys
#
# # 获取的是解释器后参数值
# print(sys.argv)

# p343 打印进度条
import time

# res = ''
# for i in range(100):
#     res += '#'
#     time.sleep(0.5)
#     print('\r[%-50s]' % res, end='')

# 模拟文件下载, 打印进度条
recv_size = 0
total_size = 3333333
while recv_size < total_size:
    time.sleep(0.001)
    recv_size += 1024

    percent = recv_size / total_size
    res = int(50 * percent) * '#'
    print('\r[%-50s] %d%%' % (res, int(percent * 100)), end='')


# p344 shutil模块
# 作用: 处理文件、文件夹、压缩包
import shutil
# 1、文件拷贝

# 需要用open来打开文件
shutil.copyfileobj(open('old.txt', 'r'), open('new.txt', 'w'))

# 直接指定文件就可以
shutil.copyfile('f1.txt', 'f2.txt')

# 仅拷贝权限, 内容、组、用户均不变
shutil.copymode('f1.txt', 'f2.txt')

# 仅拷贝状态的信息, 包括: mode bits, atime, mtime, flags
shutil.copystat('f1.txt', 'f2.txt')

# 拷贝文件, 并且把原文件的权限的等信息也全部拷贝
shutil.copy('f1.txt', 'f2.txt')

# 拷贝文件和状态信息
shutil.copy2('f1.txt', 'f2.txt')

# 递归拷贝文件夹 ignore参数是再拷贝过程中忽略以py结尾和以tmp开头的文件和文件夹
shutil.copytree('folder1', 'folder2', ignore=shutil.ignore_patterns('*.py', 'tmp*'))

# 递归的去删除文件夹及其内容
shutil.rmtree('folder')

# 文件夹移动
shutil.move('folder1', 'folder2')