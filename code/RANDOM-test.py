import sys
from typing import Dict, List, Tuple
import tqdm
import pickle

import gym
import collections
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt
from ENV.ENV_DDRQN import Environ

def setup_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
setup_seed(520)

# Q_network
class Q_net(nn.Module):
    def __init__(self, state_space=None, action_space=None):
        super(Q_net, self).__init__()

        assert state_space is not None, "None state_space input: state_space should be selected."
        assert action_space is not None, "None action_space input: action_space should be selected."

        self.Linear1 = nn.Linear(state_space, 32)
        self.Linear2 = nn.Linear(32, 32)
        self.Linear3 = nn.Linear(32, action_space)
        self.N_action = action_space

    def forward(self, x):
        x = F.relu(self.Linear1(x))
        x = F.relu(self.Linear2(x))
        return self.Linear3(x)

    def sample_action(self, obs, epsilon):
        if random.random() < epsilon:
            return random.randint(0, self.N_action - 1)
        else:
            return self.forward(obs).argmax().item()


class ReplayBuffer:
    """A simple numpy replay buffer."""

    def __init__(self, obs_dim: int, size: int, batch_size: int = 64):
        self.obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.next_obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size], dtype=np.float32)
        self.rews_buf = np.zeros([size], dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.max_size, self.batch_size = size, batch_size
        self.ptr, self.size, = 0, 0

    def put(
            self,
            obs: np.ndarray,
            act: np.ndarray,
            rew: float,
            next_obs: np.ndarray,
            done: bool,
    ):
        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self) -> Dict[str, np.ndarray]:
        idxs = np.random.choice(self.size, size=self.batch_size, replace=False)
        return dict(obs=self.obs_buf[idxs],
                    next_obs=self.next_obs_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs])

    def __len__(self) -> int:
        return self.size
    
    def clear(self):
        self.ptr, self.size, = 0, 0


def train(q_net=None, target_q_net=None, replay_buffer=None, device=None, optimizer=None, batch_size=64, learning_rate=2e-3, gamma=0.9):
    assert device is not None, "None Device input: device should be selected."

    # Get batch from replay buffer
    loss_func = nn.MSELoss()
    samples = replay_buffer.sample()

    states = torch.FloatTensor(samples["obs"]).to(device)
    actions = torch.LongTensor(samples["acts"].reshape(-1, 1)).to(device)
    rewards = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(device)
    next_states = torch.FloatTensor(samples["next_obs"]).to(device)
    dones = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(device)

    # Define loss
    q_target_max = target_q_net(next_states).max(1)[0].unsqueeze(1).detach()
    targets = rewards + gamma * q_target_max * dones
    q_out = q_net(states)
    q_a = q_out.gather(1, actions)

    # Multiply Importance Sampling weights to loss
    loss = loss_func(q_a, targets)

    # Update Network
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def save_model(model, path='default.pth'):
    torch.save(model.state_dict(), path)

"""

这里只要动作是随机的就行了,所以RANDOM-test和RANDOM有点点不一样(按照不同的代码修改的)也没关系。

"""

if __name__ == "__main__":
    # Set gym environment
    env = Environ()
    device = torch.device("cuda:0")
    batch_size = 16
    learning_rate = 2e-3
    buffer_len = int(20000)
    min_buffer_len = batch_size
    episodes = 1500
    n_task = 5
    print_per_iter = 20
    target_update_period = 1000
    steps = 0
    update_period = 40
    eps_start = 1
    eps_end = 0.05
    eps_decay = 0.98
    # tau = 1 * 1e-2
    max_step = 100
    path_pre = './RANDOM'
    path = './RANDOM-test'
    
    # Create Q functions
    Q = []
    Q_target = []
    state_space = env.state_dim
    action_space = env.action_dim
    replay_buffer = []
    optimizer = []
    # read tasks
    with open('tasks.pkl', 'rb') as f:
        tasks = pickle.load(f)
    train_task = tasks[:200]
    test_task = tasks[100:]
    train_task_list = list(range(len(train_task)))
    test_task_list = list(range(len(test_task)))
    

    epsilon = eps_start
    train_score_list = []
    test_score_list = []
    lmz = []
    energy = []
    hop = []
    suc = []
    for i in range(episodes):
        train_score_list.append(0)
        test_score_list.append(0)
        lmz.append(0)
        energy.append(0)
        hop.append(0)
        suc.append(0)

    Trange1 = tqdm.tqdm(range(n_task))
    for N_TASK in Trange1:
        task_idx = np.random.choice(test_task_list)
        task = test_task[task_idx]
        Trange2 = tqdm.tqdm(range(episodes))
        epsilon = eps_start
        for i_step in Trange2:
            # ------------------------------------------ testing ------------------------------------------
            obs = env.reset(task)
            
            test_score = 0
            episode_energy = 0
            episode_jump = 0
            episode_suc = 0
            a = []
            env.t_jammer = - env.jammer_start

            for t in range(max_step):
                a = [0 for _ in range(env.n_ch)]
                for j in range(env.n_ch):
                    a[j] = np.random.randint(0, env.action_dim)
                next_obs, r, done, _ = env.step(a)
                # for j in range(env.n_ch):
                #     replay_buffer[j].put(obs[j], a[j], r.sum() / env.n_ch, next_obs[j], done)
                obs = next_obs

                test_score += r.sum() / env.n_ch
                e, j, s = env.reward_details()
                episode_energy += e
                episode_jump += j
                episode_suc += s
                env.clear_reward()

                # if len(replay_buffer[0]) >= min_buffer_len:
                #     if (t + 1) % update_period == 0:
                #         for j in range(env.n_ch):
                #             train(Q[j], Q_target[j], replay_buffer[j], device, optimizer=optimizer[j], batch_size=batch_size, learning_rate=learning_rate)
                if done:
                    break

            # ------------------------------------------ saving ------------------------------------------
            # Trange2.set_postfix(test_score=test_score, epsilon=epsilon * 100)
            # train_score_list.append(0)
            test_score_list[i_step] += test_score/n_task

            epsilon = max(eps_end, epsilon * eps_decay)  # Linear annealing
            # lmz.append(0)
            lmz[i_step] += 0
            # energy.append(episode_energy)
            energy[i_step] += episode_energy/n_task
            # hop.append(episode_jump)
            hop[i_step] += episode_jump/n_task
            # suc.append(episode_suc)
            suc[i_step] += episode_suc/n_task
            
        np.save(path + '/rew.npy', lmz)
        np.save(path + '/energy.npy', energy)
        np.save(path + '/hop.npy', hop)
        np.save(path + '/suc.npy', suc)

        DataFrame = pd.DataFrame([train_score_list, test_score_list], index = ['train', 'test']).T
        DataFrame.to_csv(path + '/reward.csv', index=False)