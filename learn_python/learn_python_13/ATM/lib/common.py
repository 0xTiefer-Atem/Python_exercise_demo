# 通用公共库函数
import time
from conf import settings

# 记录日志功能
def logger(msg):
    with open(settings.LOG_PATH, mode='at',
              encoding='utf-8') as log_f:
        log_f.write('%s %s\n' % (time.strftime('%Y-%m-%d %H:%M:%S'), msg))