# p339 os模块
import os
# 查询路径下有哪些文件夹
# res = os.listdir(r'F:\Jetbrains\PythonSoftWare\PythonProjects\learn_python')
# print(res)

# 查看文件大小
# size = os.path.getsize(r'F:\Jetbrains\PythonSoftWare\PythonProjects\learn_python\learn_python_14\ptyhon_random_01.py')
# print(size)

# 小作业: 统计指定文件夹的大小



# 运行shell脚本或者python命令
# os.system(r'dir c:\\')

# 环境变量
# 规定key与value都必须为字符串
# print(os.environ)

# p341 os.path
# 返回当前文件的绝对路径
# print(os.path.abspath(__file__))

# 切分文件路径
# 根据不同系统切分方式不同
# res = os.path.split(r'C:\\a\\b\\c\\d.txt')
# print(res)  # ('C:\\\\a\\\\b\\\\c', 'd.txt')

# 获取文件夹名称
print(os.path.dirname(r'C:\\a\\b\\c\\d.txt'))

# 获取文件名称
print(os.path.basename(r'C:\\a\\b\\c\\d.txt'))