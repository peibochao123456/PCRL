import random
import numpy as np
from collections import namedtuple, deque
import torch
from model import Net


Transition = namedtuple('Transition', ('state', 'next_state', 'action', 'reward', 'mask'))

small_epsilon = 0.0001
device = torch.device("cpu")

class Memory_With_TDError(object):

    def __init__(self,opt):
        self.memory = deque(maxlen=opt.buffer_size)
        self.memory_probabiliy = deque(maxlen=opt.buffer_size)
        self.capacity = opt.buffer_size

        self.alpha = opt.alpha

    def push(self, state, next_state, action, reward, mask):
        """Saves a transition."""
        if len(self.memory) > 0:
            # 对于新存入的经验，指定为当前最大的优先级
            max_probability = max(self.memory_probabiliy)
        else:
            # 如果是第一条经验，初始化优先级为 small_epsilon
            max_probability = small_epsilon
        # 添加到经验池中
        self.memory.append(Transition(state, next_state, action, reward, mask))
        # 添加优先级
        self.memory_probabiliy.append(max_probability)

    # 抽样
    def sample(self, batch_size, agent, beta):
        # 优先级总和
        probability_sum = sum(self.memory_probabiliy)
        # p 列表包含每个优先级相对于总和的比例（每条经验的概率）
        p = [probability / probability_sum for probability in self.memory_probabiliy]
        # 根据每条经验的概率进行采样，返回对应样本的索引
        indexes = np.random.choice(np.arange(len(self.memory)), batch_size, p=p)
        # 根据索引来寻找对应的经验
        transitions = [self.memory[idx] for idx in indexes]
        # 采样的经验的概率
        transitions_p = [p[idx] for idx in indexes]
        batch = Transition(*zip(*transitions))

        # 消除偏差
        weights = [pow(self.capacity * p_j, -beta) for p_j in transitions_p]
        weights = torch.Tensor(weights).to(device)
        # print(weights)
        weights = weights / weights.max()
        # print(weights)

        td_error = agent.get_td_error(batch)

        td_error_idx = 0
        for idx in indexes:
            self.memory_probabiliy[idx] = pow(abs(td_error[td_error_idx]) + small_epsilon, self.alpha).item()
            # print(pow(abs(td_error[td_error_idx]) + small_epsilon, alpha).item())
            td_error_idx += 1

        return batch, weights

    def __len__(self):
        return len(self.memory)