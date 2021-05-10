# p368 正则表达式
import re

# 匹配字母下划线
# res = re.findall('\w', 'abc123_*()-=')
# print(res)

# 匹配非字母下划线
# res = re.findall('\W', 'abc123_*()-= ')
# print(res)

# 匹配任意空白字符 [\t\r\n\f]
# res = re.findall('\s', 'ab\fc123\n_*()\t-=')
# print(res)

# 匹配非空字符 [\t\r\n\f]
# res = re.findall('\S', 'ab\fc123\n_*()\t-=')
# print(res)

# 匹配任意数字 [0-9]
# res = re.findall('\d', 'ab\fc123\n_*()\t-=')
# print(res)

# 匹配非数字字符
# res = re.findall('\D', 'ab\fc123\n_*()\t-=')
# print(res)

# 匹配字符串开始 只匹配开头
# res = re.findall('\Axxx', ' xxxab\fcxxx123\n_*()\t-=')
# print(res)

# 匹配字符串结束, 如果存在换行, 只匹配到换行前结束的字符
# res = re.findall('xxx\Z', ' xxxab\fcxxx123\n_*()\t-=xxx')
# print(res)

# 匹配字符串开头
# res = re.findall('^xxx', 'xxxab\fcxxx123\n_*()\t-=xxx')
# print(res)

# 匹配字符串的末尾
# res = re.findall('xxx$', 'xxxab\fcxxx123\n_*()\t-=xxx')
# print(res)

# 重复匹配：| . | * | ? | .* | .*? | + | {n,m} |
# .: 匹配任意个字符, 除了\n之外, 指定re.DOTALL才能匹配换行符
# print(re.findall('a.b', 'a1b'))  # ['a1b']

# *: 左侧字符重复零次或无穷次
# print(re.findall('ab*', 'a ab abbbb bbbbbbbbbbb'))  # ['a', 'ab', 'abbbb']

# +: 左侧字符重复至少1次或无穷次
# print(re.findall('ab+', 'a ab abbbb bbbbbbbbbbb'))  # ['ab', 'abbbb']

# ?: 左侧字符重复0次或1次
# print(re.findall('ab?', 'a ab abbbb bbbbbbbbbbb'))

# {n,m}: 控制左侧字符重复n次到m次
# {0,} -> *
# {1,} -> +
# {0,1} -> ?
# {n} -> 单独一个n代表左侧字符只出现n次

# print(re.findall('ab{2}', 'abbb'))  # ['abb']
# print(re.findall('ab{2,4}', 'abbb'))  # ['abb']
# print(re.findall('ab{1,}', 'abbb'))  # 'ab{1,}' ===> 'ab+'
# print(re.findall('ab{0,}', 'abbb'))  # 'ab{0,}' ===> 'ab*'
# print(re.findall('ab{2}c', 'ac abbc abbbbc abbbc'))  # ['abbc']

# 练习: 找出所有的整数和小数
# print(re.findall('\d+\.?\d*', "asdfasdf123as1.13dfa12adsf1asdf3"))

# [] -> 匹配指定字符中的其中一个
# 在[]内都是普通字符
# print(re.findall('a[1*-]b', 'a1b a*b a-b'))  # []内的都为普通字符了，且如果-没有被转意的话，应该放到[]的开头或结尾
# print(re.findall('a[^1*-]b', 'a1b a*b a-b a=b'))  # []内的^代表的意思是取反，所以结果为['a=b']
# print(re.findall('a[0-9]b', 'a1b a*b a-b a=b'))  # 0-9
# print(re.findall('a[a-z]b', 'a1b a*b a-b a=b aeb'))  # 小写
# print(re.findall('a[a-zA-Z]b', 'a1b a*b a-b a=b aeb aEb'))  # 大小写
