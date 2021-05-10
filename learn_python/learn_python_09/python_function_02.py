# 作业
# 1、 编写文件修改功能，调用函数时用户传入三个参数
#     修改路径，要修改的内容，修改后的内容，就可完成文件内容的修改
import os


def opt_file(file_path, old_content, new_content):
    if os.path.exists(file_path):
        with open(r'{}'.format(file_path), mode='rt', encoding='utf-8') as f1, \
                open('./doc/test1.txt', mode='wt', encoding='utf-8') as f2:
            for line in f1:
                res = line.replace(old_content, new_content)
                f2.write(res)
    else:
        print('文件不存在')
    os.remove(file_path)
    os.rename('./doc/test1.txt', file_path)


opt_file('./doc/test.txt', 'tank3', 'aaa')
