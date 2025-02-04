import osmnx as ox
import numpy as np
from math import sin, cos, sqrt, atan2, radians

"""
Utility model and help functions.
"""
# 返回路网的图和节点列表
def prepare_graph(my_graph_file, my_node_file):
    """
    loads graph and nodes prepared in load_graph.py
    """
    my_graph = ox.load_graphml(my_graph_file)
    with open(my_node_file, "r") as file:
        my_node_list = eval(file.readline())
    return my_graph, my_node_list

# 计算单个充电站的cost
def cost_single(my_node, my_station, my_node_dict, my_cost_dict):
    """
    calculate the social cost for one station
    """
    s_pos, s_x, s_dict = my_station[0], my_station[1], my_station[2]
    # check if distance has to be calculated
    if s_pos[0] in my_node_dict[my_node[0]]:
        distance = my_node_dict[my_node[0]][s_pos[0]]
    else:
        distance = haversine(s_pos, my_node)
        my_node_dict[my_node[0]][s_pos[0]] = distance
    # check if cost has to be calculated
    try:
        _a = my_cost_dict[my_node[0]]
    except KeyError:
        my_cost_dict[my_node[0]] = {}
    if s_pos[0] in my_cost_dict[my_node[0]]:
        cost_node = my_cost_dict[my_node[0]][s_pos[0]]
    else:
        cost_travel = alpha * distance / VELOCITY
        cost_boring = (1 - alpha) / distance * (s_dict["W_s"] + 1 / s_dict["service rate"])
        cost_node = weak_demand(my_node) * (cost_travel + cost_boring)
        my_cost_dict[my_node[0]][s_pos[0]] = cost_node
    return cost_node, my_node_dict, my_cost_dict

# 选择cost最小的充电站
def station_seeking(my_plan, my_node_list, my_node_dict, my_cost_dict):
    """
    output station assignment: Each node gets assigned the charging station with minimal social cost
    """
    for the_node in my_node_list:
        cost_list = [cost_single(the_node, my_station, my_node_dict, my_cost_dict) for my_station in my_plan]
        costminindex = cost_list.index(min(cost_list))
        chosen_station = my_plan[costminindex]
        s_pos = chosen_station[0]
        the_node[1]["charging station"] = s_pos[0]
        the_node[1]["distance"] = my_node_dict[the_node[0]][s_pos[0]]
    return my_node_list, my_node_dict, my_cost_dict


################################################################################################
# 计算充电站的土地成本和充电器成本之和   fee(s)
def installment_fee(my_station):
    """
    returns cost to install the respective chargers at that position
    """
    s_pos, s_x, s_dict = my_station[0], my_station[1], my_station[2]
    charger_cost = np.sum(INSTALL_FEE * s_x)
    fee = price_parkingplace * s_pos[1]['estate price'] + charger_cost
    s_dict["fee"] = fee  # [fee] = €
    return my_station

# 计算充电站的总充电功率之和  C(S)
def charging_capability(my_station):
    """
    returns the summed up charging capability of the CS
    返回CS的充电能力总和
    """
    s_pos, s_x, s_dict = my_station[0], my_station[1], my_station[2]
    total_capacity = np.sum(CHARGING_POWER * s_x)
    s_dict["capability"] = total_capacity  # [capability] = kw
    return my_station

# 减弱后的节点的充电需求  dem_weak(v)
def weak_demand(my_node):
    return my_node[1]["demand"] * (1 - 0.1 * my_node[1]["private CS"])

# 计算充电站的影响半径    r(s)
def influence_radius(my_station):
    """
    gives the radius of the nodes whose charging demand the CS could satisfy
    """
    s_pos, s_x, s_dict = my_station[0], my_station[1], my_station[2]
    total_capacity = s_dict["capability"]
    radius_s = RADIUS_MAX * 1 / (1 + np.exp(-total_capacity / (100 * capacity_unit)))
    s_dict["radius"] = radius_s  # [radius] = m
    return my_station

# 计算充电站和节点之间的距离  dist(s,v)
def haversine(s_pos, my_node):
    """
    yields the approximate distance of two GPS points, middle computational cost
    """
    # 充电站s的经纬度坐标
    lon1, lat1 = s_pos[1]['x'], s_pos[1]['y']
    R_earth = 6372800  # approximate radius of earth. [R_earth] = m
    # 节点v的经纬度坐标
    lon2, lat2 = my_node[1]['x'], my_node[1]['y']
    dlon = radians(lon2 - lon1)
    dlat = radians(lat2 - lat1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = R_earth * c  # [distance] = m
    if distance < 0.1:  # to avoid ZeroDivisionError
        distance = 0.1
    return distance

# 计算充电站的影响半径内的节点数
def nodes_covered(my_station, my_node_list):
    """
    yields the number of nodes within the influence radius of the station
    """
    s_pos, s_x, s_dict = my_station[0], my_station[1], my_station[2]
    radius_s = s_dict["radius"]
    I_1 = sum([1 if haversine(s_pos, my_node) <= radius_s else 0 for my_node in my_node_list])
    return I_1

# 计算单个节点的benefit   benefit(v)
def node_coverage(my_plan, my_node):
    """
    yields the number of nodes within the influence radius of the station
    """
    I_1, I_2 = 0, 0
    priv_CS = my_node[1]["private CS"]
    for my_station in my_plan:   # 计算一个节点的影响半径范围内充电站的个数
        s_pos, s_x, s_dict = my_station[0], my_station[1], my_station[2]
        radius_s = s_dict["radius"]
        distance = haversine(s_pos, my_node)
        if distance <= radius_s:
            I_1 += 1
    for ith in range(I_1):
        I_2 += 1 / (ith + 1)
    single_benefit = I_2 * (1 - 0.1 * priv_CS)   # 计算单个节点的benefit
    return single_benefit

# 在充电的单位时间间隔内产生达到S的电动汽车总数    D(S)
def total_number_EVs(my_station, my_node_list):
    """
    yields total number of EVs coming to S in a unit time interval for charging
    """
    s_pos, s_x, s_dict = my_station[0], my_station[1], my_station[2]
    D_s = sum([1 / my_node[1]["distance"] * weak_demand(my_node) if my_node[1]["charging station"] == s_pos[0]
               else 0 for my_node in my_node_list])
    s_dict["D_s"] = D_s  # dimensionless
    return my_station

# 充电站s的服务率   μ(s)
def service_rate(my_station):
    """
    returns how many cars can be served within one hour
    返回一小时内可以服务的汽车数量
    """
    s_pos, s_x, s_dict = my_station[0], my_station[1], my_station[2]
    s_dict["service rate"] = s_dict["capability"] / BATTERY  # [service rate] = 1/h
    return my_station

# 计算期望的排队等待时间  waiting time(s)
def W_s(my_station):
    """
    returns the expected value of waiting time
    """
    s_pos, s_x, s_dict = my_station[0], my_station[1], my_station[2]
    tau_s = 1 / s_dict["service rate"]   # RuntimeWarning: divide by zero encountered in double_scalars
    rho_s = s_dict["D_s"] * tau_s * time_unit  # dimensionless (shortened away)
    if rho_s >= 1:
        my_W_s = my_inf
        s_dict["W_s"] = my_W_s
    else:
        my_W_s = rho_s * tau_s / (2 * (1 - rho_s))  # W_s = expected waiting time at S, [W_s] = h
        s_dict["W_s"] = my_W_s
    return my_station

# 设置充电站的参数
def s_dictionnary(my_station, my_node_list):
    """
    returns the dictionnary for the station
    """
    my_station = installment_fee(my_station)  # 安装费用
    my_station = charging_capability(my_station)  # 总充电功率
    my_station = influence_radius(my_station)  # 影响半径
    my_station = total_number_EVs(my_station, my_node_list)  # 到达S的电动汽车总数
    my_station = service_rate(my_station)  # 服务率
    my_station = W_s(my_station)  # 期望排队等待时间
    return my_station


# SCORE #####################################################################
# 计算一个计划的benefit   benefit(p)
def social_benefit(my_plan, my_node_list):
    """
    returns the social benefit of the charging plan (our definition of benefit)
    """
    my_benefit = 0
    for my_node in my_node_list:
        I3 = node_coverage(my_plan, my_node)
        my_benefit += I3
    my_benefit /= len(my_node_list)
    return my_benefit

# 计算所有车辆的估计行驶时间  travel time(p)
def travel_cost(my_node_list):
    """ yields the estimated travel time of all vehicles """
    my_cost_travel = sum([my_node[1]["distance"] * weak_demand(my_node) / VELOCITY for my_node in my_node_list])
    return my_cost_travel

# 在给定充电计划的CS能力的情况下，得出总充电时间  charging time(p)
def charging_time(my_plan):
    """
    yields the total charging time given the capability of the CS of the charging plan
    """
    my_charg_time = sum([my_station[2]["D_s"] / my_station[2]["service rate"] for my_station in my_plan])
    return my_charg_time / time_unit

# 返回充电计划的平均总等待时间（所有充电站的平均等待时间）  waiting time(p)
def waiting_time(my_plan):
    """
    returns the average total waiting time of the charging plan
    """
    my_wait_time = sum([my_station[2]["D_s"] * my_station[2]["W_s"] for my_station in my_plan])
    return my_wait_time / time_unit

# 计算分配计划的cost  cost(p)
def social_cost(my_plan, my_node_list):
    """
    returns the social cost, i.e. the negative side of the charging plan
    """
    cost_travel = travel_cost(my_node_list)  # dimensionless
    charg_time = charging_time(my_plan)  # dimensionless
    wait_time = waiting_time(my_plan)  # dimensionless
    cost_boring = charg_time + wait_time  # dimensionless
    my_social_cost = alpha * cost_travel + (1 - alpha) * cost_boring
    return my_social_cost

# 计算现存基础设置的一些属性，benefit，旅行时间,充电时间，等待时间，cost
def existing_score(my_existing_plan, my_node_list):
    """
    computes the score of the existing infrastructure
    """
    my_benefit = social_benefit(my_existing_plan, my_node_list)  # 计算现存基础设施的benefit
    travel_time = travel_cost(my_node_list)  # dimensionless  #
    charg_time = charging_time(my_existing_plan)  # dimensionless
    wait_time = waiting_time(my_existing_plan)
    cost_boring = charg_time + wait_time  # dimensionless
    my_cost = alpha * travel_time + (1 - alpha) * cost_boring
    return my_benefit, my_cost, charg_time, wait_time, travel_time

# 返回当前计划的一些参数
def norm_score(my_plan, my_node_list, norm_benefit, norm_charg, norm_wait, norm_travel):
    """
    same as score, but normalised.
    """
    my_score = -my_inf
    if not my_plan:
        return my_score
    benefit = social_benefit(my_plan, my_node_list) / norm_benefit
    cost_travel = travel_cost(my_node_list) / norm_travel  # dimensionless
    charg_time = charging_time(my_plan) / norm_charg  # dimensionless
    wait_time = waiting_time(my_plan) / norm_wait  # dimensionless
    cost = (alpha * cost_travel + (1 - alpha) * (charg_time + wait_time)) / 3
    my_score = my_lambda * benefit - (1 - my_lambda) * cost
    return my_score, benefit, cost, charg_time, wait_time, cost_travel

# 返回最终结果，即score   score(p)
def score(my_plan, my_node_list):
    """
    returns the final result, i.e., the social score
    """
    my_score = -my_inf
    benefit = 0
    cost = 0
    if not my_plan:
        return my_score, benefit, cost
    benefit = social_benefit(my_plan, my_node_list)  # dimensionless
    cost = social_cost(my_plan, my_node_list)
    my_score = my_lambda * benefit - (1 - my_lambda) * cost
    return my_score, benefit, cost


# Constraints checks ############################################################################
# 约束检查 对于计划内的充电站，一个充电站的总充电器数量不能大于K
def station_capacity_check(my_plan):
    """
    check if number of stations exceed capacity
    """
    for my_station in my_plan:
        s_x = my_station[1]
        if sum(s_x) > K:
            print("Error: More chargers at the station than admitted: {} chargers".format(sum(s_x)))

# 检查是否超出预算
def installment_cost_check(my_plan, my_basic_cost):
    """
    check if instalment costs exceed budget
    """
    total_inst_cost = sum([my_station[2]["fee"] for my_station in my_plan]) - my_basic_cost
    if total_inst_cost > BUDGET:
        print("Error: Maximal BUDGET for installation costs exceeded.")

# 检查一个充电站是否分配多个节点
def control_charg_decision(my_plan, my_node_list):
    for my_node in my_node_list:
        station_sum = sum([1 for my_station in my_plan if my_node[1]["charging station"] == my_station[0]])
        if station_sum > 1:
            print("Error: More than one station is assigned to a node.")

# 检查等待时间是否超出限制
def waiting_time_check(my_plan):
    """
    check that wiating time is bounded
    """
    for my_station in my_plan:
        s_dict = my_station[2]
        if s_dict["W_s"] == my_inf:
            print("Error: Waiting time goes to infinity.")

# 检查是否满足所有约束条件
def constraint_check(my_plan, my_node_list, basic_cost):
    """
    test if solution satisfies all constraints
    """
    installment_cost_check(my_plan, basic_cost)
    control_charg_decision(my_plan, my_node_list)
    station_capacity_check(my_plan)
    waiting_time_check(my_plan)


# Parameters ########################################################
alpha = 0.4
my_lambda = 0.5

K = 8  # maximal number of chargers at a station
RADIUS_MAX = 1000  # [radius_max] = m  最大影响半径
INSTALL_FEE = np.array([300, 750, 28000])  # fee per installing a charger of type 1, 2 or 3. [fee] = $   每种类型充电器的费用
CHARGING_POWER = np.array([7, 22, 50])  # [power] = kW, rounded   每种类型充电器的功率
BATTERY = 85  # battery capacity, [BATTERY] = kWh   每辆车需充电的功率

BUDGET = 5 * 10 ** 5  # [B] = €   总预算
price_parkingplace = 200 * 3.5 * 2  # in €

time_unit = 1  # [time_unit] = h, introduced for getting the units correctly
capacity_unit = 1  # [cap_unit] = kW, introduced for getting the units correctly
VELOCITY = 23 * 1000  # based on m per hour, but here dimensionless

my_inf = 10 ** 6
my_dis_inf = 10 ** 7
