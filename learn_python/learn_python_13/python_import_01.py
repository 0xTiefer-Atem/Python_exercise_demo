# 一个py文件有两种用途

# 1、当作程序运行

# 2、当作模块导入


print(__name__)
# 1、当py文件被运行时，__name__的值为'__main__'
# 2、当py文件被当作模块导入时，__name__的值为该文件的名
if __name__ == '__main__':
    # 运行方法
    print("文件被执行")
    pass
else:
    # 被当作模块导入是做的事情
    print("文件被导入")
    pass
