"""
此处存放核心业务逻辑代码
"""
from db import db_hanlder
from lib import common
import time

# 若能进入该函数, 证明用户已经登录, 能获取到当前用户名
login_user = None


# 注册功能
def register():
    print("注册功能执行中")
    while True:
        user_name = input("请输入用户名 (输入q退出): ").strip()
        if user_name == 'q':
            break

        # 1)先校验用户是否存在
        # 涉及数据操作: 查看数据
        # 传入当前用户名,查看当前用户是否存在
        user_data = db_hanlder.select(user_name)

        # 2)若不存在,让用户重新输入
        if user_data:
            print("当前用户已存在,请重新输入")
            continue

        user_pwd = input("请输入密码: ").strip()
        re_user_pwd = input("请再次输入密码: ").strip()

        # 3)检验两次密码是否一致
        if user_pwd == re_user_pwd:
            # 4)将当前用户写入文件中
            db_hanlder.save(user_name, user_pwd)
            print("{} 用户注册成功".format(user_name))
            break
        else:
            print("两次密码不一致,请重新输入")


# 登陆功能
def login():
    print("登录功能执行中")
    while True:
        user_name = input("请输入用户名 (输入q退出): ").strip()

        if user_name == 'q':
            break

        # 1)先校验用户是否存在
        user_data = db_hanlder.select(user_name)

        # 2)若不存在,让用户重新输入
        if not user_data:
            print("当前用户不存在,请重新输入")
            continue

        user_pwd = input("请输入密码: ").strip()

        # 3)检验密码是否正确
        if user_pwd == user_data[1]:

            # 4)登陆成功后记录登陆状态
            global login_user
            login_user = user_name

            print("{} 用户登录成功".format(user_name))
            break
        else:
            print("密码错误, 登陆失败")


# 充值功能
@common.login_auth
def charge():
    print("充值功能执行中")
    while True:
        # 1) 让用户输入充值金额
        balance = input("请输入充值金额: ").strip()

        # 2) 判断用户输入的是否为数字
        if not balance.isdigit():
            print("请输入正确的金额!")
            continue

        balance = int(balance)
        if balance < 0:
            print("请输入大于零的金额")
            continue

        # 3) 修改当前用户的金额
        # 3.1) 获取当前用户的金额, 拼接成旧数据
        user_name, user_pwd, user_balance = db_hanlder.select(login_user)
        old_user_data = '{}:{}:{}'.format(user_name, user_pwd, user_balance)

        # 3.3) 修改当前用户金额, 做加钱操作
        user_balance = int(user_balance)
        user_balance += balance

        # 3.3) 拼接修改后的当前用户数据
        new_user_data = '{}:{}:{}'.format(user_name, user_pwd, user_balance)

        # 3.5) 调用修改数据功能
        db_hanlder.update(old_user_data, new_user_data)
        print("当前用户 {} 充值金额 [{}元] 成功".format(login_user, balance))

        # 记录充值日志
        now_time = time.strftime('%Y-%m-%d %X')
        log_data = f'时间: {now_time} 用户: {login_user} 充值金额: {balance}'
        common.append_log(log_data)
        break


# 阅读小说功能
@common.login_auth
def read_novel():
    """
    1、写该功能之前, 先将小说数据存放在story_class.txt文件中
    2、先将story_class.txt文件中数据获取出来, 解析成字典类型
    :return:
    """
    # 获取story_class.txt文件中的字典数据
    story_dic = db_hanlder.get_all_story()

    # 判断story_txt文件中是否有小说数据
    if not story_dic:
        print("当前没有小说, 请联系管理员上传")
        return
    while True:
        # 1) 打印小说种类信息
        print("""
                ===========小说类型选择===========
                            0、玄幻
                            1、都市
                            2、实用
                ===========   end    ===========
        """)
        # 2) 让用户选择小说类型
        choice1 = input("请输入小说类型编号: ").strip()

        # 3) 判断当前用户选择的编号是否存在
        if choice1 not in story_dic:
            print("输入有误, 请重新输入")
            continue

        # 4) 获取当前小说的类型中的所有小说数据
        novel_dic = story_dic.get(choice1)

        # 5) 打印当前类型的所有小说, 让用户选择
        for number, novel_list in novel_dic.items():
            name, price = novel_list
            print(f"小说编号 [{number}] 小说名字 [{name}] 小说价格 [{price}]")

        # 6) 让用户选择需要购买的小说
        while True:
            choice2 = input("需要购买的的小说编号: ").strip()
            if choice2 not in novel_dic:
                print("输入有误, 请重新输入")
                continue

            name, price = novel_dic.get(choice2)

            # 7) 让用户输入y选择是否购买商品
            choice3 = input(f'当前选择的小说为 [{name}], 价格为 [{price}], 请输入y进行购买, 或者退出: ').strip()

            # 8) 判断用户输入的是否是Y
            if choice3 == 'y':

                # 9) 校验当前用户的金额是否大于等于小说单价
                # 9.1) 获取用户当前金额
                user_name, user_pwd, user_balance = db_hanlder.select(login_user)

                # 9.2) 判断金额
                # 当前用户金额
                user_balance = int(user_balance)

                # 小说单价
                price = int(price)
                if user_balance < price:
                    print('金额不足, 请及时充值')
                    break

                # 10) 开始扣费
                # 10.1) 拼接用户修改前的数据
                old_user_data = f'{user_name}:{user_pwd}:{user_balance}'
                # 10.2) 拼接用户修改后的数据
                user_balance = user_balance - price
                new_user_data = f'{user_name}:{user_pwd}:{user_balance}'
                db_hanlder.update(old_user_data, new_user_data)

                print('当前小说买成功, 自动打开小说阅读')

                # 11) 调用获取小说的内容
                novel_data = db_hanlder.show_novel(name)

                print(
                    f"""
                    ====
                    {novel_data}
                    ====
                    """
                )

                # 记录购买日志
                now_time = time.strftime('%Y-%m-%d %X')
                log_data = f'时间: {now_time} 用户: {login_user} 消费金额: {price}'
                common.append_log(log_data)
        break


# 创建函数字典
fun_dic = {
    '0': register,
    '1': login,
    '2': charge,
    '3': read_novel
}


# 启动函数
def run():
    while True:
        print("""
        ===== 小说阅读器欢迎您 =====
                0、账号注册
                1、账号登录
                2、充值功能
                3、阅读小说
                q、退出
        =====       end      =====
        """)
        choice = input("请输入功能编号: ").strip()
        if choice == 'q':
            print("退出小说阅读软件")
            break

        # 判断用户输入的编号是否在函数字典中
        if choice not in fun_dic:
            print("当前编号有误, 请重新输入")
            continue
        fun_dic[choice]()
