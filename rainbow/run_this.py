import argparse
import torch
from memory import Memory_With_TDError
from model import Net,DQN_Agent
from tensorboardX import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import env_plus as ev

# 超参数设置
parser = argparse.ArgumentParser()

# parser.add_argument('--env_name', type=str, default='CartPole-v1', help='环境的名称')
parser.add_argument('--buffer_size', type=int, default=int(1e6), help='经验池的容量')

parser.add_argument('--net_width', type=int, default=128, help='Hidden net width')
parser.add_argument('--lr', type=float, default=0.001, help='学习率')
parser.add_argument('--gamma', type=float, default=0.99, help='Discounted Factor')
parser.add_argument('--epsilon', type=float, default=1.0, help='c-贪婪策略')
parser.add_argument('--target_update', type=int, default=10000, help='目标网络的更新频率')
parser.add_argument('--initial_exploration', type=int, default=1000, help='目标网络的更新频率')
parser.add_argument('--batch_size', type=int, default=32, help='lenth of sliced trajectory')
parser.add_argument('--log_interval', type=int, default=10, help='日志记录')

parser.add_argument('--beta_init', type=float, default=0.1, help='beta for PER')
parser.add_argument('--alpha', type=float, default=0.5, help='alpha for PER')

opt = parser.parse_args()



def main():
    os.environ['PYTHONASHSEED'] = '0'
    # 设置随机种子
    seed = 1
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    np.random.seed(seed)
    random.seed(seed)
    # Instantiate the env
    location = "Toy_Example"  # take a location of your choice
    graph_file = "../Graph/ShenZhen/ShenZhen.graphml"
    node_file = "../Graph/ShenZhen/SZ_nodes_extended.txt"
    plan_file = "../Graph/ShenZhen/SZ_existingplan.pkl"

    env = ev.StationPlacement(graph_file, node_file, plan_file)
    log_dir = "per_tmp_Toy_Example/"
    modelname = "best_model_" + location + "_"

    # 获取环境状态的维度
    opt.state_dim = env.observation_space.shape[0]  # 129
    # 获取动作的维度
    opt.action_dim = env.action_space.n  # 5

    # print(opt.state_dim)
    # print(opt.action_dim)

    # CPU运算
    opt.device = torch.device("cpu")

    # 实例化经验池
    replay_buffer = Memory_With_TDError(opt=opt)
    # 实例化DQN
    agent = DQN_Agent(n_states=opt.state_dim,n_hidden=opt.net_width,n_actions=opt.action_dim,learning_rate=opt.lr,gamma=opt.gamma,epsilon=opt.epsilon,target_update=opt.target_update,device=opt.device)

    beta = opt.beta_init

    score_list = []
    running_score = 0
    steps = 0
    loss = 0
    writer = SummaryWriter('logs')

    for e in range(200000):
        done = False

        score = 0
        # 初始化状态
        state = env.reset()[0]
        state = torch.Tensor(state).to(opt.device).unsqueeze(0)

        while not done:
            steps += 1

            action = agent.take_action(state)
            # 更新环境
            next_state, reward, done, _,_ = env.step(action)
            next_state = torch.Tensor(next_state).unsqueeze(0)
            mask = 0 if done else 1

            reward = reward if not done or score == 499 else -1
            action_one_hot = np.zeros(opt.action_dim)
            action_one_hot[action] = 1
            # 添加经验池
            replay_buffer.push(state, next_state, action_one_hot, reward, mask)

            # 累加奖励
            score += reward
            # 更新当前状态
            state = next_state

            if steps > opt.initial_exploration:
                agent.epsilon -= 0.00005
                agent.epsilon = max(agent.epsilon, 0.1)
                beta += 0.00005
                beta = min(beta,1)

                # 从经验池中随机抽样作为训练集
                batch,weights = replay_buffer.sample(batch_size=opt.batch_size,agent=agent, beta=beta)
                # 网络更新
                loss = agent.update(batch=batch,weights=weights)

                if steps % opt.target_update == 0:
                    agent.update_target_model()

        score = score if score == 500.0 else score + 1
        running_score = 0.99 * running_score + 0.01 * score
        score_list.append(running_score)
        if e % opt.log_interval == 0:
            print('{} episode | score: {:.2f} | epsilon: {:.5f} | loss: {:.2f}'.format(
                e, running_score, agent.epsilon,loss))
            writer.add_scalar('log/score', float(running_score), e)
            writer.add_scalar('log/loss', float(loss), e)

        # if running_score > 200:
        #     break
    plt.plot(list(range(len(score_list))), score_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('PCRL on {}'.format(opt.env_name))
    plt.show()


if __name__ == '__main__':
    main()


