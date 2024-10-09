import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

# -------------------------------------- #
# 构造深度学习网络，输入状态s，得到各个动作的reward
# -------------------------------------- #

class Net(nn.Module):

    def __init__(self,n_states,n_hidden,n_actions,num_layers=1):
        super(Net,self).__init__()
        self.lstm = nn.LSTM(input_size=n_states, hidden_size=64, num_layers=num_layers, batch_first=True)
        self.fc1 = nn.Linear(64,n_hidden)
        self.fc2 = nn.Linear(n_hidden,n_actions)

    def forward(self,x):
        x = x.unsqueeze(1)
        lstm_out, _ = self.lstm(x)
        # 取序列的最后一个时间步的输出
        final_hidden_state = lstm_out[:, -1, :]

       # print(final_hidden_state.shape)

        x = F.relu(self.fc1(final_hidden_state))

        x =self.fc2(x)

        return x

# -------------------------------------- #
# 构造强化学习代理模型
# -------------------------------------- #

class DQN_Agent:

    def __init__(self,n_states, n_hidden, n_actions,learning_rate, gamma, epsilon,target_update, device):
        # 属性分配
        self.n_states = n_states  # 状态的特征数
        self.n_hidden = n_hidden  # 隐含层个数
        self.n_actions = n_actions  # 动作数
        self.learning_rate = learning_rate  # 训练时的学习率
        self.gamma = gamma  # 折扣因子，对下一状态的回报的缩放
        self.epsilon = epsilon  # 贪婪策略，有1-epsilon的概率探索
        self.target_update = target_update  # 目标网络的参数的更新频率
        self.device = device  # 在CPU计算
        # 计数器，记录迭代次数
        self.count = 0
        # 构建2个神经网络，相同的结构，不同的参数
        # 实例化训练网络  [b,4]-->[b,2]  输出动作对应的奖励
        self.q_net = Net(self.n_states, self.n_hidden, self.n_actions)
        # 实例化目标网络
        self.target_q_net = Net(self.n_states, self.n_hidden, self.n_actions)
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        # 优化器，更新训练网络的参数
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=self.learning_rate)

    # 选择行动
    def take_action(self,state):
        # 如果小于该值就取最大的值对应的索引
        if np.random.random() > self.epsilon:
            # 前向传播获取该状态对应的动作的reward
            actions_value = self.q_net(state)
            # 获取reward最大值对应的动作索引
            action = actions_value.argmax().item()  # int
        # 如果大于该值就随机探索
        else:
            action = np.random.randint(self.n_actions)
        return action

    def get_td_error(self,batch):
        states = torch.stack(batch.state).squeeze(1)
        next_states = torch.stack(batch.next_state).squeeze(1)
        actions = torch.Tensor(batch.action).float()
        rewards = torch.Tensor(batch.reward)
        masks = torch.Tensor(batch.mask)

        pred = self.q_net(states).squeeze(1)
        next_pred =self.target_q_net(next_states).squeeze(1)
        pred = torch.sum(pred.mul(actions), dim=1)
        target = rewards + masks * self.gamma * next_pred.max(1)[0]
        td_error = pred - target.detach()

        return td_error

    # 网络训练
    def update(self, batch,weights):  # 传入经验池中的batch个样本

        td_error = self.get_td_error(batch)
        loss = pow(td_error,2) * weights
        loss = loss.mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss

    # 更新目标网络的参数
    def update_target_model(self):
        self.target_q_net.load_state_dict(self.q_net.state_dict())



