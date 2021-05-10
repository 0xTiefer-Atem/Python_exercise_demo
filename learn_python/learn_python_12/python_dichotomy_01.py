# 二分法
# 需求: 有一个按照从小到大排列的数字列表
#       根据输入的数字,判断该数字在不在列表内, 在返回下标, 不在返回-1
num_list = [0, 3, 5, 6, 8, 12, 34, 78, 90]


# 方法一:
# def binary_search(num):
#     low = 0
#     high = len(num_list) - 1
#     while low < high:
#         mid = (low + high) // 2
#         if num == num_list[mid]:
#             return mid
#         if num > num_list[mid]:
#             low = mid
#         if num < num_list[mid]:
#             high = mid
#     return -1
#
#
# print(binary_search(6))
# print(binary_search(100))


# 方法二: 递归
def binary_search(find_num, l):
    mid_index = len(l) // 2
    mid_val = l[mid_index]
    if find_num > mid_val:
        l = l[mid_index + 1:]
        binary_search(find_num, l)
    if find_num < mid_val:
        l = l[:mid_index]
        binary_search(find_num, l)
    else:
        print("find it , index: {}".format(mid_index))
