# p354 configparser
# 来加载特殊格式的配置文件
import configparser

# 1、读取配置文件
config = configparser.ConfigParser()
config.read('./doc/test1.ini')

# 2、获取里面所有的sections
print(config.sections())

# 3、获取某个section下的配置参数
print(config.options('section2'))

# 4、获取kv键值对
print(config.items('section2'))

# 5、获取指定的值
print(config.get('section2', 'k11'))