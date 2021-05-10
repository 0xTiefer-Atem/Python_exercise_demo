"""
此处存放固定的配置信息
"""

import os

# 获取项目根目录
BASE_PATH = os.path.dirname(os.path.dirname(__file__))

# 获取db目录路径
DB_PATH = os.path.join(BASE_PATH, 'db')

# db.txt的根目录
DB_TXT_PATH = os.path.join(DB_PATH, 'db.txt')

# story_class.txt文件目录路径
STORY_PATH = os.path.join(DB_PATH, 'story_class.txt')

# 拼接小说文件目录路径
NOVEL_DIR = os.path.join(DB_PATH, 'novels')

# 日志文件路径
LOG_PATH = os.path.join(
    BASE_PATH, 'log', 'log.txt'
)
