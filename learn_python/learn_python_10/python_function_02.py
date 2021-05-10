# 函数对象应用实例
# 结合ATM取款机
def login():
    print("登录")


def transfer():
    print("转账功能")


def check_balance():
    print("查询余额")


def withdraw():
    print("提现")


fun_dic = {
    '1': ['登录', login],
    '2': ['转账', transfer],
    '3': ['查询余额', check_balance],
    '4': ['提现', withdraw]
}

while True:
    for k in fun_dic:
        print(k, fun_dic[k][0])

    command = input("请输入命令: ")
    if not command.isdigit():
        print("必须是以上编号,憨批")
        continue

    if command in fun_dic:
        if fun_dic[command][0] is None:
            break
        else:
            fun_dic[command][1]()
    else:
        print("没有该命令")
