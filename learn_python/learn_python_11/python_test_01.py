# 编写计数器功能，要求调用一次在原有的基础上加一
# 需要用到闭包+nonlocal
#       核心功能:
#         def counter():
#           x+=1
#           return x


def outer(x):
    def counter():
        nonlocal x
        x += 1
        return x

    return counter


inner = outer(0)
counter = inner
print(counter())
print(counter())
print(counter())
print(counter())
print(counter())