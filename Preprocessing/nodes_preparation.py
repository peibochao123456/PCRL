import osmnx as ox
import pickle
import numpy as np
import evaluation_framework as ef


# 返回路网的图和节点列表
def prepare_graph(my_graph_file, my_node_file):
    """
    loads graph and nodes prepared in load_graph.py
    """
    my_graph = ox.load_graphml(my_graph_file)
    with open(my_node_file, "r") as file:
        my_node_list = eval(file.readline())
    return my_graph, my_node_list

# 每个节点的需求
def modify_demand_dict(my_demand_matrix):
    """
    adapt demand to the number of nodes in each grid cell
    """
    for i in range(len(my_demand_matrix)):
        for j in range(len(my_demand_matrix[0])):
            if grid_density[i][j] > 0:
                my_demand_matrix[i][j] = my_demand_matrix[i][j] / grid_density[i][j]
            else:
                my_demand_matrix[i][j] = 0
            my_demand_matrix[i][j] += demand_min
            if my_demand_matrix[i][j] >= 250:
                my_demand_matrix[i][j] = 250
    demand_max = np.amax(my_demand_matrix)
    my_demand_matrix /= demand_max
    my_demand_matrix = np.round(my_demand_matrix, 3)
    return my_demand_matrix

# 计算一个节点充电需求
def charging_demand(my_node):
    """
    calculates the charging demand for each node
    """
    if my_node[1]["row"] is None:
        my_node[1]["demand"] = np.mean(demand_matrix)
    else:
        # demand per unit time interval, therefore dimensionless
        my_node[1]["demand"] = demand_matrix[my_node[1]["row"]][my_node[1]["column"]]

# 计算社会效率的上限()  最大的benefit
def social_efficiency_upper_bound(my_node, my_node_list):
    """
    calculate the social efficiency for each node
    """
    priv_CS = my_node[1]["private CS"]
    I1_max = 0  # dimensionless
    for other_node in my_node_list:
        # calculate distance with haversine approximation
        if ef.haversine(my_node, other_node) <= ef.RADIUS_MAX:
            I1_max += 1
    my_node[1]["I1_max"] = I1_max
    # delta_benefit每个路口的充电需求
    delta_benefit = I1_max * (1 - 0.1 * priv_CS)
    delta_benefit /= 100  # does not matter as we scale here
    upper_bound = ef.my_lambda * delta_benefit / my_node[1]["estate price"]
    return upper_bound

if __name__ == '__main__':
    location = "ShenZhen"
    # 路网文件路径
    graph_file = "../Graph/ShenZhen/ShenZhen.graphml"
    # 节点的文件路径
    node_file = "../Graph/ShenZhen/SZ_node_list.txt"

    # 获取路网graph和节点node_list
    graph, node_list = prepare_graph(graph_file, node_file)
    # print(node_list)

    # 获取网格的结点密度
    with (open("../Graph/ShenZhen/SZ_grid_density.pkl", "rb")) as f:
        grid_density = pickle.load(f)

    # 获取充电需求数据
    with (open("../Graph/ShenZhen/SZ_grid_demand.pkl","rb")) as f:
        demand = pickle.load(f)

    # 获取节点的土地成本
    with (open("../Graph/ShenZhen/SZ_estateprice.pkl", "rb")) as f:
        estate_matrix = pickle.load(f)

    """
    Preparation of the nodes. 1) Demand 
    """
    demand_min = 0.05
    demand_matrix = demand
    demand_matrix = modify_demand_dict(demand_matrix)
    for node in node_list:
        charging_demand(node)

    """
    2.) Estate price
    """
    for node in node_list:
        if node[1]["row"] is None:
            node[1]["estate price"] = np.mean(estate_matrix)
        else:
            node[1]["estate price"] = estate_matrix[node[1]["row"]][node[1]["column"]]
    print(node_list)

    """
    3.) Private Charging stations  （先不考虑）
    """
    for node in node_list:
        if node[1]["row"] is None:
            node[1]["private CS"] = 0
        else:
            node[1]["private CS"] = 0
    print(node_list)

    """
    4.)  Maximum of nodes covered if this node becomes a charging station.
         如果此节点成为充电站，则覆盖的最大节点数。
    """
    # 计算每个节点的benefit的上限
    for node in node_list:
        node[1]["upper bound"] = social_efficiency_upper_bound(node, node_list)
    print(node_list)

    """
    5.) Existing charging infrastructure
    """
    existing_plan = []  # 假设为空

    with open("../Graph/ShenZhen/SZ_nodes_extended.txt",'w') as file:
        file.write(str(node_list))

    print(node_list)

    pickle.dump(existing_plan, open("../Graph/ShenZhen/SZ_existingplan.pkl", "wb"))




