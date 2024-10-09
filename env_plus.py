from typing import Optional
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pickle
from random import choice
from math import ceil
import itertools
import evaluation_framework as ef

"""
Custom environment   自定义环境
"""
# 对于不同容量的配置去发现最便宜的其中一个
def prepare_config():   # my_config_dict  不同功率所需要的充电器各类型的数量
    """
    we prepare the power capacities of the different charging configuration and to find the cheapest ones
    """
    N = len(ef.CHARGING_POWER)    #N=3
    urn = list(range(0, ef.K + 1)) * N
    config_list = []
    for combination in itertools.combinations(urn, N):
        config_list.append(list(combination))   # 获得随机的（2，5，6） 三个数字随机在0-9内

    my_config_dict = {}  # 字典  ‘容量’：[2,4,1](充电器的数量)
    for config in config_list:
        if np.sum(config) > ef.K:
            continue
        else:
            capacity = np.sum(ef.CHARGING_POWER * config)
            if capacity in my_config_dict.keys():
                # check if we have found a better configuration for the same capacity
                if np.sum(ef.INSTALL_FEE * config) < np.sum(ef.INSTALL_FEE * my_config_dict[capacity]):   # 如果容量一样，但是成本更少，则更新config
                    my_config_dict[capacity] = config
            else:
                my_config_dict[capacity] = config
    # if we have a cheaper price at more capacity, we will use that configuration even if less capacity is required
    key_list = sorted(list(my_config_dict.keys()))   # 对容量进行排序 key_list
    for index, key in enumerate(key_list):
        cost_list = [np.sum(ef.INSTALL_FEE * my_config_dict[my_key]) for my_key in key_list[index:]]
        best_cost_index = cost_list.index(min(cost_list)) + index
        best_config = my_config_dict[key_list[best_cost_index]]
        my_config_dict[key] = best_config
    return my_config_dict
# key_list
# [0, 7, 14, 21, 22, 28, 29, 35, 36, 42, 43, 44, 49, 50, 51, 56, 57, 58, 64, 65, 66, 71, 72, 73, 78, 79, 80, 85, 86, 87, 88, 92, 93, 94, 95, 99, 100, 101, 102, 107, 108, 109, 110, 114, 115, 116, 117, 121, 122, 123, 124, 128, 129, 130, 131, 132, 135, 136, 137, 138, 139, 142, 143, 144, 145, 146, 150, 151, 152, 154, 157, 158, 159, 160, 161, 164, 165, 166, 167, 171, 172, 173, 174, 176, 178, 179, 180, 182, 185, 186, 187, 188, 189, 193, 194, 195, 200, 201, 202, 204, 207, 208, 210, 214, 215, 216, 217, 221, 222, 223, 228, 229, 230, 232, 236, 238, 243, 244, 245, 250, 251, 257, 258, 260, 264, 266, 271, 272, 273, 279, 286, 288, 294, 300, 301, 307, 314, 316, 322, 329, 344, 350, 357, 372, 400]
#  my_config_dict
# {122: [2, 5, 0], 172: [0, 8, 0], 222: [0, 6, 2], 272: [1, 3, 4], 322: [0, 1, 6], 372: [0, 1, 7], 22: [0, 1, 0], 72: [1, 3, 0], 194: [0, 7, 1], 244: [1, 4, 3], 294: [0, 2, 5], 344: [0, 2, 6], 44: [0, 2, 0], 94: [1, 4, 0], 144: [2, 6, 0], 266: [0, 3, 4], 316: [0, 3, 5], 66: [0, 3, 0], 116: [1, 5, 0], 166: [0, 8, 0], 216: [1, 5, 2], 88: [0, 4, 0], 138: [1, 6, 0], 188: [1, 6, 1], 238: [0, 4, 3], 288: [0, 4, 4], 110: [0, 5, 0], 160: [1, 7, 0], 210: [0, 5, 2], 260: [0, 5, 3], 132: [0, 6, 0], 182: [0, 6, 1], 232: [0, 6, 2], 154: [0, 7, 0], 204: [0, 7, 1], 176: [0, 8, 0], 50: [1, 2, 0], 100: [2, 4, 0], 150: [0, 7, 0], 200: [0, 7, 1], 250: [0, 5, 3], 300: [1, 2, 5], 350: [0, 0, 7], 400: [0, 0, 8], 0: [0, 0, 0], 201: [0, 7, 1], 251: [0, 5, 3], 301: [1, 2, 5], 51: [1, 2, 0], 101: [2, 4, 0], 151: [0, 7, 0], 273: [1, 3, 4], 73: [1, 3, 0], 123: [2, 5, 0], 173: [0, 8, 0], 223: [0, 6, 2], 95: [1, 4, 0], 145: [2, 6, 0], 195: [0, 7, 1], 245: [1, 4, 3], 117: [1, 5, 0], 167: [0, 8, 0], 217: [1, 5, 2], 139: [1, 6, 0], 189: [1, 6, 1], 161: [1, 7, 0], 57: [2, 2, 0], 107: [0, 5, 0], 157: [1, 7, 0], 207: [0, 5, 2], 257: [0, 5, 3], 307: [0, 3, 5], 357: [1, 0, 7], 7: [1, 0, 0], 129: [0, 6, 0], 179: [0, 6, 1], 229: [0, 6, 2], 279: [0, 4, 4], 329: [1, 1, 6], 29: [1, 1, 0], 79: [2, 3, 0], 80: [2, 3, 0], 130: [0, 6, 0], 180: [0, 6, 1], 230: [0, 6, 2], 102: [2, 4, 0], 152: [0, 7, 0], 202: [0, 7, 1], 124: [2, 5, 0], 174: [0, 8, 0], 146: [2, 6, 0], 64: [0, 3, 0], 114: [1, 5, 0], 164: [0, 8, 0], 214: [1, 5, 2], 264: [0, 3, 4], 314: [0, 3, 5], 14: [2, 0, 0], 136: [1, 6, 0], 186: [1, 6, 1], 236: [0, 4, 3], 286: [0, 4, 4], 36: [2, 1, 0], 86: [0, 4, 0], 208: [0, 5, 2], 258: [0, 5, 3], 58: [2, 2, 0], 108: [0, 5, 0], 158: [1, 7, 0], 109: [0, 5, 0], 159: [1, 7, 0], 131: [0, 6, 0], 71: [1, 3, 0], 121: [2, 5, 0], 171: [0, 8, 0], 221: [0, 6, 2], 271: [1, 3, 4], 21: [0, 1, 0], 143: [2, 6, 0], 193: [0, 7, 1], 243: [1, 4, 3], 43: [0, 2, 0], 93: [1, 4, 0], 215: [1, 5, 2], 65: [0, 3, 0], 115: [1, 5, 0], 165: [0, 8, 0], 87: [0, 4, 0], 137: [1, 6, 0], 187: [1, 6, 1], 78: [2, 3, 0], 128: [0, 6, 0], 178: [0, 6, 1], 228: [0, 6, 2], 28: [1, 1, 0], 85: [0, 4, 0], 135: [1, 6, 0], 185: [1, 6, 1], 35: [2, 1, 0], 92: [1, 4, 0], 142: [2, 6, 0], 42: [0, 2, 0], 99: [2, 4, 0], 49: [1, 2, 0], 56: [2, 2, 0]}

# 对于一个充电站s，先求s所需要的功率，然后求可满足功率的最小值和对于充电器类型的数量，函数返回充电器类型的数量
def initial_solution(my_config_dict, my_node_list, s_pos):
    """
    get the initial solution for the charging configuration
    """
    W = 0  # minimum capacity constraint
    radius = 50
    for my_node in my_node_list:
        if ef.haversine(s_pos, my_node) <= radius:
            W += ef.weak_demand(my_node)
    W = ceil(W) * ef.BATTERY   # w表示在充电站s的radius区域内需要充电的车所需总功率
    key_list = sorted(list(my_config_dict.keys()))   # 对总功率排序
    for key in key_list:
        if key > W:
            break
    # 遇见第一个可满足s的功率
    best_config = my_config_dict[key]  # [0,2,1] 三种类型的数量
    return best_config

# 计算每个节点的影响半径内充电站的个数，并计算benefit
def coverage(my_node_list, my_plan):
    """
    see which nodes are covered by the charging plan
    """
    for my_node in my_node_list:
        cover = ef.node_coverage(my_plan, my_node)
        my_node[1]["covered"] = cover  # covered 代表每个节点的benefit

# 在未分配充电站的节点，选择covered最小的节点
def choose_node_new_benefit(free_list):
    """
    pick location which the smallest coverage
    """
    upbound_list = [my_node[1]["covered"] for my_node in free_list]
    pos_minindex = upbound_list.index(min(upbound_list))
    chosen_node = free_list[pos_minindex]
    return chosen_node

# 选择充电需求最大的节点
def choose_node_bydemand(free_list):
    """
    pick location with highest weakened demand
    """
    demand_list = [my_node[1]["demand"] * (1 - 0.1 * my_node[1]["private CS"]) for my_node in free_list]
    chosen_index = demand_list.index(max(demand_list))
    chosen_node = free_list[chosen_index]
    return chosen_node

# 在方案内分配充电站的节点，选择coverage最小的节点
def anti_choose_node_bybenefit(my_node_list, my_plan):
    """
    choose station with the least coverage
    """
    plan_list = [station[0][0] for station in my_plan]
    my_occupied_list = [node for node in my_node_list if node[0] in plan_list]
    if not my_occupied_list:
        return None
    upbound_list = [node[1]["upper bound"] for node in my_occupied_list]
    pos_minindex = upbound_list.index(min(upbound_list))
    remove_node = my_occupied_list[pos_minindex]
    plan_index = plan_list.index(remove_node[0])
    remove_station = my_plan[plan_index]
    return remove_station

# 对于一个充电站，计算充电时间和排队等待时间，返回二者之和
def support_stations_hilfe(station):
    charg_time = station[2]["D_s"] / station[2]["service rate"]
    wait_time = station[2]["D_s"] * station[2]["W_s"]
    neediness = (wait_time + charg_time)
    return neediness

# 在分配计划内的充电站的节点，选择充电时间和排队等待时间之和最大的节点，
def support_stations(my_plan, free_list):
    """
    choose a station which needs support due to highest waiting + charging time
    """
    cost_list = [support_stations_hilfe(station) for station in my_plan]
    if not cost_list:
        chosen_node = choose_node_bydemand(free_list)
    else:
        index = cost_list.index(max(cost_list))
        station_sos = my_plan[index]
        if sum(station_sos[1]) < ef.K:    # 充电站s已经存在的充电器数量小于K
            chosen_node = station_sos[0]
        else:                              # 如果不小于K，选择距离s最近的未分配节点
            # look for nearest node that could support the station
            dis_list = [ef.haversine(station_sos[0], my_node) for my_node in free_list]
            min_index = dis_list.index(min(dis_list))
            chosen_node = free_list[min_index]
    return chosen_node

# 计划类
class Plan:
    def __init__(self, my_node_list, my_node_dict, my_cost_dict, my_plan_file):  # 为计划plan创建实例
        with (open(my_plan_file, "rb")) as f:
            self.plan = pickle.load(f)
        self.plan = [ef.s_dictionnary(my_station, my_node_list) for my_station in self.plan]   # 计算计划内充电站的各个属性
        my_node_list, _, _ = ef.station_seeking(self.plan, my_node_list, my_node_dict, my_cost_dict)
        # update the dictionnary
        self.plan = [ef.s_dictionnary(my_station, my_node_list) for my_station in self.plan]
        # 现存设施的一些参数
        self.norm_benefit, self.norm_cost, self.norm_charg, self.norm_wait, self.norm_travel = \
            ef.existing_score(self.plan, my_node_list)
        self.existing_plan = self.plan.copy()
        self.existing_plan = [s[0] for s in self.existing_plan]

    def __repr__(self):
        return "The charging plan is {}".format(self.plan)

    def add_plan(self, my_station):
        self.plan.append(my_station)

    def remove_plan(self, my_station):
        self.plan.remove(my_station)
# 从充电站减少一个充电器，给出预算反馈，并且求出充电器的类型
    def steal_column(self, stolen_station, my_budget):
        """
        steal a charger from the station, give budget back and check which charger type has been stolen
        """
        my_budget += stolen_station[2]["fee"]  # 预算增加  先加上充电站的所有费用
        station_index = self.plan.index(stolen_station)
        # we choose the most expensive charging column
        if stolen_station[1][2] > 0:
            self.plan[station_index][1][2] -= 1
            config_index = 2
        elif stolen_station[1][1] > 0:
            self.plan[station_index][1][1] -= 1
            config_index = 1
        else:
            self.plan[station_index][1][0] -= 1
            config_index = 0
        if sum(stolen_station[1]) == 0:   # 充电站没有剩余充电器
            # this means we remove the entire stations as it only has one charger
            self.remove_plan(stolen_station)
        else:  # 充电站还有充电器
            # the station remains, we only steal one charging column
            ef.installment_fee(stolen_station)    # 修改充电站的属性
            my_budget -= stolen_station[2]["fee"]  #减去更新后充电站的费用
        return my_budget, config_index


class Station:
    def __init__(self):
        self.s_pos = None    # 充电站的位置
        self.s_x = None      # 充电站内各个类型充电器的数量
        self.s_dict = {}     # 充电站的相关属性
        self.station = [self.s_pos, self.s_x, self.s_dict]

    def __repr__(self):
        return "This station is {}".format(self.station)

    def add_position(self, my_node):
        self.station[0] = my_node   # s_pos  节点的位置信息

    def add_chargers(self, my_config):
        self.station[1] = my_config  # 各种类型充电器的数量

    def establish_dictionnary(self, node_list):
        self.station = ef.s_dictionnary(self.station, node_list)  # 更新充电站的属性

# 交互的环境(自定义环境)
class StationPlacement(gym.Env):
    """Custom Environment that follows gym interface"""
    node_dict = {}
    cost_dict = {}

    def __init__(self, my_graph_file, my_node_file, my_plan_file):
        super(StationPlacement, self).__init__()
        _graph, self.node_list = ef.prepare_graph(my_graph_file, my_node_file)
        self.plan_file = my_plan_file
        self.node_list = [self.init_hilfe(my_node) for my_node in self.node_list]
        self.game_over = None
        self.budget = None
        self.plan_instance = None
        self.plan_length = None
        self.row_length = 5
        self.best_score = None
        self.best_plan = None
        self.best_node_list = None
        self.schritt = None
        self.config_dict = None
        # new action space including all charger types
        self.action_space = spaces.Discrete(5)   # 5个离散的动作
        shape = (self.row_length + len(ef.CHARGING_POWER)) * len(self.node_list) + 1
        self.observation_space = spaces.Box(low=-1, high=1, shape=(shape,), dtype=np.float16)

    # 将环境状态重置为初始状态，在episode开始时调用，它返回一个观察结果和一个带有附加信息的字典
    def reset(self,seed=None, options=None):
        # 后加的
        super().reset(seed=seed)
        """
        Reset the state of the environment to an initial state
        """
        self.budget = ef.BUDGET
        self.game_over = False
        self.plan_instance = Plan(self.node_list, StationPlacement.node_dict, StationPlacement.cost_dict,
                                  self.plan_file)
        self.best_score, _, _, _, _, _ = ef.norm_score(self.plan_instance.plan, self.node_list,
                                                       self.plan_instance.norm_benefit, self.plan_instance.norm_charg,
                                                       self.plan_instance.norm_wait, self.plan_instance.norm_travel)
        self.plan_length = len(self.plan_instance.existing_plan)   # 目前计划的长度
        self.schritt = 0
        self.best_plan = []
        self.best_node_list = []
        self.config_dict = prepare_config()
        coverage(self.node_list, self.plan_instance.plan)
        obs = self.establish_observation()
        info = {}
        return obs,info

    def get_info(self):
        return self.game_over

    def init_hilfe(self, my_node):
        StationPlacement.node_dict[my_node[0]] = {}  # prepare node_dict
        StationPlacement.cost_dict[my_node[0]] = {}
        my_node[1]["charging station"] = None
        my_node[1]["distance"] = None
        return my_node
# 建立观察矩阵
    def establish_observation(self):
        """
        Build observation matrix
        """
        row_length = self.row_length + len(ef.CHARGING_POWER)    # 5+3
        width = row_length * len(self.node_list) + 1
        obs = np.zeros((width,))           # [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]  一维数组
        for j, node in enumerate(self.node_list):   # j是索引值  node是结点的信息
            i = j * row_length
            obs[i + 0] = node[1]['x']
            obs[i + 1] = node[1]['y']
            obs[i + 2] = node[1]['demand']
            obs[i + 3] = node[1]['estate price']
            obs[i + 4] = node[1]['private CS']
            for my_station in self.plan_instance.plan:
                if my_station[0][0] == node[0]:
                    for e in range(len(ef.CHARGING_POWER)):
                        index = 5 + e
                        obs[i + index] = my_station[1][e]
                    break
        obs[-1] = self.budget   # 最后一个格子存储预算
        obs = np.divide(obs, ef.BUDGET)
        obs = np.asarray(obs, dtype=self.observation_space.dtype)
        return obs
# 整个充电站的预算调整
    def budget_adjustment(self, my_station):
        inst_cost = my_station[2]["fee"]
        if self.budget - inst_cost > 0:
            # if we have enough money, we build the station
            self.budget -= inst_cost
        else:
            self.game_over = True  # 预算不够 游戏结束
# 充电器的预算调整
    def budget_adjustment_small(self, config_index):
        if self.budget - ef.INSTALL_FEE[config_index] > 0:
            # if we have enough money, we build the charger
            self.budget -= ef.INSTALL_FEE[config_index]
        else:
            self.game_over = True

    def prepare_score(self):
        """
        We have to make a loop to reorganise the station assignment
        """
        for j in range(2):
            self.node_list, _, _ = ef.station_seeking(self.plan_instance.plan, self.node_list, StationPlacement.node_dict,
                                             StationPlacement.cost_dict)
            for i in range(len(self.plan_instance.plan)):
                self.plan_instance.plan[i] = ef.total_number_EVs(self.plan_instance.plan[i], self.node_list)
                self.plan_instance.plan[i] = ef.W_s(self.plan_instance.plan[i])
            j += 1
# 被调用以对环境采取行动，它返回下一个观察结果、立即奖励、新状态是否是最终状态（情节完成）、是否达到最大时间步数（情节人为完成）以及附加信息
    def step(self, my_action):
        """
        Perform a step in the episode
        """
        chosen_node, free_list_zero, config_index, action = self._control_action(my_action)
        if chosen_node in free_list_zero:
            # build new station  建立新的充电站
            default_config = initial_solution(self.config_dict, self.node_list, chosen_node)
            station_instance = Station()
            station_instance.add_position(chosen_node)
            station_instance.add_chargers(default_config)
            station_instance.establish_dictionnary(self.node_list)
            # Step: Control budget
            self.budget_adjustment(station_instance.station)
            if not self.game_over:
                self.plan_instance.add_plan(station_instance.station)
        else:
            # add column to existing CS
            station_index = None
            for station in self.plan_instance.plan:
                if station[0][0] == chosen_node[0]:
                    station_index = self.plan_instance.plan.index(station)
                    break
            # Step: Control budget
            self.budget_adjustment_small(config_index)
            if not self.game_over:
                self.plan_instance.plan[station_index][1][config_index] += 1
        # Step: calculate reward
        reward = self.evaluation()
        coverage(self.node_list, self.plan_instance.plan)
        obs = self.establish_observation()
        # episode end conditions
        if len(self.plan_instance.plan) == len(self.node_list):
            self.game_over = True
        self.schritt += 1
        if self.schritt >= len(self.node_list) / 2:
            self.game_over = True
        # if self.game_over:
        #     print("Best score {}.".format(self.best_score))
        info = {}
        return obs, reward, self.game_over, False,info

    def station_config_check(self, my_station):
        """
        no more than K chargers are allowed at the station
        """
        capacity = True
        if sum(my_station[1]) >= ef.K:
            capacity = False
        return capacity
# 采取行动，有三种可能性
    def _control_action(self, chosen_action):
        """
        we have three possibilities here: either build a new station, add a charger to an exisiting station or move a
        charger from an exisiting station to a station in need
        """
        my_action = chosen_action
        config_index = None
        full_station_list = [s[0][0] for s in self.plan_instance.plan if self.station_config_check(s)   # 恰好有K的充电器的充电站
                             is False]  # these are the stations with exactly K chargers
        station_list = [s[0][0] for s in self.plan_instance.plan]  # all charging stations              # 所有充电站
        occupied_list = [node for node in self.node_list if node[0] not in full_station_list and node[0] in  # 充电器数量小于K的充电站
                         station_list]  # nodes with non-full stations
        free_list = [node for node in self.node_list if node[0] not in station_list]  # nodes without stations  # 空的节点 没有设置充电站
        if 0 <= my_action <= 1:            #建造新的充电站
            # build
            if my_action == 0:
                chosen_node = choose_node_new_benefit(free_list)    # 选择覆盖率最低的节点
            else:
                chosen_node = choose_node_bydemand(free_list)       # 选择充电需求最大的节点
        elif 2 <= my_action <= 3:             #增加充电器
            # add column to existing station                   # 增加已有充电站内的充电器
            config_index = 1
            if len(occupied_list) == 0:                       # 如果所有充电站的充电器数量都达最大值，则选择空的节点
                chosen_node = choice(free_list)
            else:
                if my_action == 2:
                    chosen_node = choose_node_new_benefit(occupied_list)
                else:
                    chosen_node = choose_node_bydemand(occupied_list)
        else:                      # 移动充电站内的充电器
            # move station
            steal_plan = [s for s in self.plan_instance.plan if s[0] not in self.plan_instance.existing_plan]
            # we can not steal from the existing charging plan
            stolen_station = anti_choose_node_bybenefit(self.node_list, steal_plan)
            if stolen_station is None:
                # only necessary if we take this action in the very beginning
                chosen_node = choice(free_list)
            else:
                self.budget, config_index = self.plan_instance.steal_column(stolen_station, self.budget)
                chosen_node = support_stations(self.plan_instance.plan, free_list)
        return chosen_node, free_list, config_index, my_action
# 计算reward
    def evaluation(self):
        """
        Calculate the reward
        """
        reward = 0
        self.prepare_score()
        new_score, _, _, _, _, _ = ef.norm_score(self.plan_instance.plan, self.node_list,
                                                 self.plan_instance.norm_benefit, self.plan_instance.norm_charg,
                                                 self.plan_instance.norm_wait, self.plan_instance.norm_travel)
        new_score = max(new_score, -25)  # if negative score
        if new_score - self.best_score > 0:
            reward += (new_score - self.best_score)
            # avoid jojo learning
            self.best_score = new_score
            self.best_plan = self.plan_instance.plan.copy()
            self.best_node_list = self.node_list.copy()
        return reward

    # 它允许可视化代理的行为
    def render(self, mode='human', close=False):
        """
        Render the environment to the screen
        """
        print(f'Score is: {self.best_score}')
        print(f'Number of stations in charging plan: {len(self.plan_instance.plan)}')
        return self.best_node_list, self.best_plan


if __name__ == '__main__':
    location = "Toy_Example"
    graph_file = "Graph/" + location + "/" + location + ".graphml"
    node_file = "Graph/" + location + "/nodes_extended_" + location + ".txt"
    plan_file = "Graph/" + location + "/existingplan_" + location + ".pkl"
    env = StationPlacement(graph_file, node_file, plan_file)
    # It will check your custom environment and output additional warnings if needed
    # 检查环境是否遵循Gym
    # check_env(env)
    # print(check_env(env))   # 环境遵循
