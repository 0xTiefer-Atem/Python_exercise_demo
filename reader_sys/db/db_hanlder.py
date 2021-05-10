"""
用于数据库操作
"""

from conf import settings
import os


# 查看数据
def select(user_name):
    """
    - 接收用户输入的用户名
    - 若该用户存在, 则返回当前用户的所有数据
    - 若不存在, 则返回None
    :param user_name:
    :return:
    """

    with open(settings.DB_TXT_PATH, mode='rt', encoding='utf-8') as account_file:
        # 获取db.txt文件中的每一行数据
        for line in account_file:
            # 判断接收过来的用户名是否存在db.txt文件中
            if user_name in line:
                # 若用户存在,则在当前行中提取该用户的所有数据
                user_data = line.strip().split(":")
                return user_data
        return None


# 保存数据
def save(user_name, user_pwd, balance=0):
    """
    :param user_name: 注册用户名
    :param user_pwd: 注册密码
    :param balance: 注册用户初始金额设置为默认值
    :return:
    """
    with open(settings.DB_TXT_PATH, 'a', encoding='utf-8') as save_account:
        save_account.write('{}:{}:{}\n'.format(user_name, user_pwd, balance))


# 更新数据
def update(old_user_data, new_user_data):
    """
    将旧数据替换成新数据
    :return:
    """
    # 1) 拼接新的文件路径
    new_path = os.path.join(
        settings.DB_PATH, 'new.txt'
    )

    # 2) 读取db.txt 文件中数据, 进行修改, 写入到新文件 new.txt中, 在更换为db.txt文件名
    with open(settings.DB_TXT_PATH, 'r', encoding='utf-8') as r_f, \
            open(new_path, 'w', encoding='utf-8') as w_f:
        # 2.1) 新旧数据替换
        all_user_data = r_f.read()
        all_user_data = all_user_data.replace(old_user_data, new_user_data)

        # 2.2) 将新数据写入新文件中
        w_f.write(all_user_data)

    # 3) 文件名修改
    os.remove(settings.DB_TXT_PATH)
    os.rename(new_path, settings.DB_TXT_PATH)


# 获取所有小说字典数据
def get_all_story():
    with open(settings.STORY_PATH, 'r', encoding='utf-8') as story_f:
        story_dic = eval(story_f.read())
        return story_dic


# 查看单本小说
def show_novel(novel_name):
    # 获取小说路径
    novel_path = os.path.join(
        settings.NOVEL_DIR, novel_name
    )

    # 打开文件获取文件数据并返回给用户展示
    with open(novel_path, 'r', encoding='utf-8') as novel_f:
        novel_data = novel_f.read()
    return novel_data