# 作业需要
import sys
import os

# 打印系统参数
print(sys.argv)

# 判断文件夹路径是否存在
if sys.argv[1]:
    # 获取需要检测的文件夹路径
    dir_path = sys.argv[1]

    if os.path.isdir(dir_path):
        file_size = os.path.getsize(dir_path)
        print(f'查看路径文件路径为 {dir_path}, 文件大小为 {file_size}')