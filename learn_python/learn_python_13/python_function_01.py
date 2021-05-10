# 函数的提示类型
# 函数名(变量名1:类型, 变量名2:类型) -> 返回值类型 :
def register(name: str, age: int, hobbies: tuple) -> int:
    print(name, age, hobbies)
    return 111


register('a', 11, (1, 2, 3))

# 打印函数的参数提示信息
print(register.__annotations__)  # {'name': <class 'str'>, 'age': <class 'int'>, 'hobbies': <class 'tuple'>, 'return': <class 'int'>}
