# 小作业
"""
用户登录接口
需求:
1、输入商户密码验证，验证通过输出   登录成功
2、可以登录不同账户
3、同一账户输错三次，锁定(锁定信息存入文件，保证程序重启，锁定信息保留)
4、编写注册，注册后登录
"""
flag = True
while flag:
    msg = '''    0 退出
    1 登录
    2 注册\n'''
    print(msg)
    cmd = input("请输入序号:  ").strip()
    if not cmd.isdigit():
        print("序号输入错误!请重新输入\n")
    else:
        cmd = int(cmd)
        if cmd < 0 or cmd > 2:
            print("序号输入有误！请重新输入\n")
        else:
            if cmd == 0:
                # 用户退出
                break
            if cmd == 1:
                username = input("请输入用户: ").strip()
                with open("../doc/account.txt", mode="at", encoding='utf-8') as account_file:
                    # 循环取出用户信息
                    for user_msg in account_file:
                        user, pwd, status = user_msg.strip().split(":")
                        status = int(status)
                        # 状态大于3，用户被锁定
                        if status > 3:
                            print("账户已锁定!\n")
                            flag = False
                            break
                        else:
                            # 用户登录
                            if user == username:
                                password = input("请输入密码: ").strip()
                                if pwd == password:
                                    print("登录成功!")
                                    flag = False
                                    break
                            else:
                                status = status + 1
                                print("登陆失败!")

            if cmd == 2:
                # 用户注册
                username = input("请输入账户: ").strip()
                password = input("请输入密码: ").strip()
                with open("../doc/account.txt", mode='at', encoding='utf-8') as account_file:
                    account_file.write("{}:{}:{}\n".format(username, password, 0))
                    print("注册成功!\n")
