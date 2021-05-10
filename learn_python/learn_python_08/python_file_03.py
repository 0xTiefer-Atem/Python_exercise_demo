# 指定文件字符集编码
'''
t文本(默认的模式)
1、读写都以str(unicode)为单位
2、文本文件
3、必须指定encoding='utf-8'
'''
with open("../doc/test.txt", mode='rt', encoding='utf-8') as file1:
    res = file1.read()
    print(res, type(res))
