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

if __name__ == "__main__":
    # Set gym environment    # Set gym     # Set gym environment


    env = Environ()
    device = torch.device("cuda:0") # DEBUG 如果参数过多，需要使用GPU
    batch_size = 32
    learning_rate = 2e-3
    buffer_len = int(20000)
    min_buffer_len = batch_size
    episodes = 4000
    print_per_iter = 20
    target_update_period = 2000
    steps = 0
    update_period = 10
    eps_start = 1.0
    eps_end = 0.05
    eps_decay = 1
    # tau = 1 * 1e-2
    max_step = 200
    path = './RANDOM'
    lmz = []
    energy = []
    hop = []
    suc = []
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
    

    for i in range(env.n_ch):
        Q.append(Q_net(state_space=state_space, action_space=action_space).to(device))
        Q_target.append(Q_net(state_space=state_space, action_space=action_space).to(device))
        Q_target[i].load_state_dict(Q[i].state_dict())
        replay_buffer.append(ReplayBuffer(state_space, size=buffer_len, batch_size=batch_size))
        optimizer.append(optim.Adam(Q[i].parameters(), lr=learning_rate))

    epsilon = eps_start
    
    train_score_list = []
    test_score_list = []

    Trange = tqdm.tqdm(range(episodes))
    for i_step in Trange:
        # ------------------------------------------ training ------------------------------------------
        task_idx = np.random.choice(train_task_list)
        task = train_task[task_idx]
        for i in range(env.n_ch):
            replay_buffer[i].clear()
        
        obs = env.reset(task)
        train_score = 0
        episode_energy = 0
        episode_jump = 0
        episode_suc = 0
        a = []
        env.t_jammer = - env.jammer_start

        for t in range(max_step):
            steps += 1
            a = [0 for _ in range(env.n_ch)]
            for j in range(env.n_ch):
                a[j] = Q[j].sample_action(torch.from_numpy(obs[j]).float().to(device), epsilon)

            next_obs, r, done, _ = env.step(a)
            for j in range(env.n_ch):
                replay_buffer[j].put(obs[j], a[j], r.sum() / env.n_ch, next_obs[j], done)
            obs = next_obs

            train_score += r.sum() / env.n_ch
            e, j, s = env.reward_details()
            episode_energy += e
            episode_jump += j
            episode_suc += s
            env.clear_reward()

            if len(replay_buffer[0]) >= min_buffer_len:
                if (t + 1) % update_period == 0:
                    for j in range(env.n_ch):
                        train(Q[j], Q_target[j], replay_buffer[j], device, optimizer=optimizer[j], batch_size=batch_size, learning_rate=learning_rate)
                if (steps + 1) % target_update_period == 0:
                    for j in range(env.n_ch):
                        for target_param, local_param in zip(Q_target[j].parameters(), Q[j].parameters()):
                            target_param.data.copy_(local_param.data)
            if done:
                break

        # ------------------------------------------ testing ------------------------------------------
        Q_params = []
        for i in range(env.n_ch):
            Q_params.append(Q[i].state_dict())

        for i in range(env.n_ch):
            replay_buffer[i].clear()

        task_idx = np.random.choice(test_task_list)
        task = test_task[task_idx]
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
                a[j] = Q[j].sample_action(torch.from_numpy(obs[j]).float().to(device), epsilon)

            next_obs, r, done, _ = env.step(a)
            for j in range(env.n_ch):
                replay_buffer[j].put(obs[j], a[j], r.sum() / env.n_ch, next_obs[j], done)
            obs = next_obs

            test_score += r.sum() / env.n_ch
            e, j, s = env.reward_details()
            episode_energy += e
            episode_jump += j
            episode_suc += s
            env.clear_reward()

            if len(replay_buffer[0]) >= min_buffer_len:
                if (t + 1) % update_period == 0:
                    for j in range(env.n_ch):
                        train(Q[j], Q_target[j], replay_buffer[j], device, optimizer=optimizer[j], batch_size=batch_size, learning_rate=learning_rate)
            if done:
                break
        for i in range(env.n_ch):
            Q[i].load_state_dict(Q_params[i])

        # ------------------------------------------ saving ------------------------------------------
        Trange.set_postfix(score=train_score, test_score=test_score, epsilon=epsilon * 100)
        train_score_list.append(train_score)
        test_score_list.append(test_score)

        epsilon = max(eps_end, eps_decay ** i_step)  # Linear annealing
        lmz.append(train_score)
        energy.append(episode_energy)
        hop.append(episode_jump)
        suc.append(episode_suc)
        
        np.save(path + '/rew.npy', lmz)
        np.save(path + '/energy.npy', energy)
        np.save(path + '/hop.npy', hop)
        np.save(path + '/suc.npy', suc)

        DataFrame = pd.DataFrame([train_score_list, test_score_list], index = ['train', 'test']).T
        DataFrame.to_csv(path + '/reward.csv', index=False)