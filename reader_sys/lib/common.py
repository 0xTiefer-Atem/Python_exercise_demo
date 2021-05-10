"""
此处存放公共功能
"""
from conf import  settings

# 登录认证装饰器
def login_auth(func):
    # 解决问题: 循环导入问题
    from core import src
    def inner(*args, **kwargs):
        if src.login_user:
            res = func(*args, **kwargs)
            return res
        else:
            print("未登录, 不允许使用特殊功能, 请先登录")
            src.login()

    return inner


# 添加日志, 应放在公共功能中
def append_log(log_data):
    # 写入日志
    with open(settings.LOG_PATH, 'a', encoding='utf-8') as log_f:
        log_f.write(log_data + '\n')
