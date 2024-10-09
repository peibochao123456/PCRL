from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.callbacks import BaseCallback
import os
import numpy as np
import torch
import random
import env_plus as ev

"""
Train the model by reinforcement learning.   
"""
# 自定义回调
class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Code from Stable Baselines3, 
    https://colab.research.google.com/github/Stable-Baselines-Team/rl-colab-notebooks/blob/sb3/monitor_training.ipynb
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param my_log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """

    def __init__(self, check_freq: int, my_log_dir: str, my_modelname: str, verbose:int=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = my_log_dir
        self.modelname = my_modelname
        self.save_path = os.path.join(self.log_dir, self.modelname)
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        # 创建文件夹
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        # 每隔check_freq步进行一次检查
        if self.n_calls % self.check_freq == 0:

            # Retrieve training reward
            x, y = ts2xy(load_results(self.log_dir), 'timesteps')
            if len(x) > 0:
                # Mean training reward over the last 100 episodes
                my_mean_reward = np.mean(y[-3:])  # 最近3个训练奖励的平均值
                # 打印当前的时间步数和最佳平均奖励以及最近一次的平均奖励
                if self.verbose > 0:
                    print("Num timesteps: {}".format(self.num_timesteps))
                    print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(
                        self.best_mean_reward, my_mean_reward))
                # 如果平均奖励大于最佳奖励  则更新最佳奖励
                if my_mean_reward > self.best_mean_reward:
                    self.best_mean_reward = my_mean_reward
                    # Example for saving best model
                    if self.verbose > 0:
                        # 打印新的最佳奖励
                        print("New best mean reward: {:.2f}".format(self.best_mean_reward))
                        # we want to make sure that the best models are not overwritten
                        new_name = self.modelname + str(self.num_timesteps)
                        if self.save_path is not None:
                            os.makedirs(self.save_path, exist_ok=True)
                        # 保存新的模型
                        self.save_path = os.path.join(self.log_dir, new_name)
                        print("Saving new best model to {}".format(self.save_path))
                    self.model.save(self.save_path)
        return True


if __name__ == '__main__':
    # pip uninstall -r requirements.txt -ypip uninstall -r requirements.txt -ypip uninstall -r requirements.txt -y set a seed for reproducibility
    os.environ['PYTHONASHSEED'] = '0'
    seed = 1
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    np.random.seed(seed)
    random.seed(seed)
    # Instantiate the env
    location = "Toy_Example"  # take a location of your choice
    graph_file = "Graph/ShenZhen/ShenZhen.graphml"
    node_file = "Graph/ShenZhen/SZ_nodes_extended.txt"
    plan_file = "Graph/ShenZhen/SZ_existingplan.pkl"

    env = ev.StationPlacement(graph_file, node_file, plan_file)
    log_dir = "tmp_Toy_Example/"
    modelname = "best_model_" + location + "_"

    # print(env.observation_space.shape[0])
    # print(env.action_space.n)

    """
    Define and train the agent
    """
    env = Monitor(env, log_dir)
    model = DQN("MlpPolicy",
                env,
                verbose=1,
                batch_size=128,
                buffer_size=10000,
                learning_rate=0.001,
                device='cpu',
                seed=seed)
    # check_freq参数（检查频率）  my_log_dir参数（保存模型的文件夹路径）和可选的verbose参数（用于控制输出信息的详细程度）
    callback = SaveOnBestTrainingRewardCallback(check_freq=400, my_log_dir=log_dir, my_modelname=modelname)
    model.learn(total_timesteps=200000, log_interval=10 ** 4, callback=callback)

