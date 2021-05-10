# 1、通用文件copy工具实现
import time

# src_file_path = input("请输入源文件路径: ")
# target_file_path = input("请输入源文件路径: ")
# with open(src_file_path, mode='rb') as src_file, \
#         open(target_file_path, mode='wb') as target_file:
#     while True:
#         file_byte = src_file.read(1024)
#         if len(file_byte) == 0:
#             break
#         target_file.write(file_byte)

# 2、实现linux tail -f 命令监控access.log文件
# 读取文件内容，循环监听
with open('../doc/access.log', mode='br') as tail_file:
    # 移动到末尾
    tail_file.seek(0, 2)
    while True:
        log = tail_file.readline()
        if log:
            print(log.decode('utf-8'))
        else:
            # 模拟延时操作
            time.sleep(0.2)
