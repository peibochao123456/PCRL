import numpy as np
import pickle

# 读取 需求 矩阵文件
with open('../../Graph/ShenZhen/SZ_grid_demand.pkl', 'rb') as file:
    demand_matrix = pickle.load(file)

# 读取密度 矩阵文件
with open('../../Graph/ShenZhen/SZ_grid_density.pkl', 'rb') as file:
    density_matrix  = pickle.load(file)

# 读取 路网 的节点
with open('../../Graph/ShenZhen/SZ_node_list.txt', "r") as file:
    node_list = eval(file.readline())


demand_list = []

for node in node_list:
    row = node[1]['row']
    col = node[1]['column']

    # 确保密度不为零，避免除零错误
    if density_matrix[row, col] > 0:
        # 更新节点需求，需求 = 需求矩阵的值 / 密度矩阵的值
        node[1]['demand'] = demand_matrix[row, col] / density_matrix[row, col]
    else:
        node[1]['demand'] = 0  # 如果密度为0，则需求设置为0或者其他默认值
    demand_list.append(node[1]['demand'])


print(demand_list)

