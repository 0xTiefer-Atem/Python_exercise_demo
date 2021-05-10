# 简单练习
# 1、使用while打印1 2 3 4 5 6   8 9 10
# count1 = 1
# while count1 <= 10:
#     if count1 == 7:
#         count1 += 1
#         continue
#     print(count1)
#     count1 += 1
#

# 2、求 1-100所有数的和
# count2 = 1
# sum1 = 0
# while count2 <= 100:
#     sum1 += count2
#     count2 += 1
# print(sum1)


# 3、输出1-100所有奇数
# count3 = 1
# while count3 <= 100:
#     if count3 % 2 != 0:
#         print(count3)
#     count3 += 1


# 4、输出1-100所有偶数
# count3 = 1
# while count3 <= 100:
#     if count3 % 2 == 0:
#         print(count3)
#     count3 += 1


# 5、求1-2+3-4+5....99的结果
count5 = 1
sum5 = 0
while count5 < 100:
    if count5 % 2 != 0:
        sum5 += count5
    else:
        sum5 -= count5
    count5 += 1
print(sum5)


# 6、用户登陆(三次机会)
# user_name = "wq"
# user_pwd = "123"
# count6 = 0
# while count6 < 3:
#     input_name = input("请输入用户名: ")
#     input_pwd = input("请输入密码: ")
#     if input_name == user_name and user_pwd == input_pwd:
#         flag = True
#         while flag:
#             cmd = input("请输入指令(退出请输入q或者quit或者'退出'): ")
#             if cmd == "q" or cmd == "quit" or cmd == "退出":
#                 flag = False
#             else:
#                 print("指令{cmd}正在执行".format(cmd=cmd))
#         break
#     else:
#         count6 += 1
#         opp = 3 - count6
#         print("账户或或密码输入错误，你还有{opp}次机会".format(opp=opp))
# else:
#     print("三次输入错误，账户已被冻结，请联系人工进行处理！")


# 7、猜年龄游戏
#    需求：允许用户最多尝试3次，三次都不对，则直接退出，若正确则打印恭喜并退出
# game_age = 18
# flag = 0
# while True:
#     if flag == 3:
#         break
#     input_age = int(input("请输入猜想的年龄: "))
#     if game_age == input_age:
#         print("恭喜回答正确!")
#         break
#     else:
#         flag += 1

# 8、猜年龄游戏
#    需求：允许用户最多尝试3次
#    每尝试3次后，询问用户是否继续猜，回答y或Y，就继续，回答n或N就退出
#    猜对直接退出

# game_age1 = 20
# flag1 = 0
# flag2 = True
# while flag2:
#     input_age = int(input("请输入猜想的年龄: "))
#     if game_age1 == input_age:
#         print("恭喜回答正确!")
#         flag2 = False
#     else:
#         flag1 += 1
#         if flag1 == 3:
#             input_flag = input("是否继续？")
#             if input_flag == "y" or input_flag == "Y":
#                 flag2 = True
#                 flag1 = 0
#             elif input_flag == "n" or input_flag == "N":
#                 flag2 = False
#                 print("游戏退出")
