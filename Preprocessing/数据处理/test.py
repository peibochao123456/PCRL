import pickle
import numpy as np

with open("../../Graph/ShenZhen/SZ_nodes_extended.txt", "r") as file:
    data = eval(file.readline())


# 提取所有的 'demand' 值
demand_list = []

for item in data:
    demand_list.append(item[1]['demand'])

# 查看读取的数据
print(demand_list)
