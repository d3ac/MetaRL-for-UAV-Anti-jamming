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

def get_context(context_list, batch=1, device=torch.device("cuda:0")):
    return torch.stack([torch.stack(random.sample(context_list, 5)) for _ in range(batch)]).to(device)

# Q_network
class Q_net(nn.Module):
    def __init__(self, state_space=None, action_space=None, context_dim=None, hidden_layers=2, hidden_size=128, latent_dim=5):
        super(Q_net, self).__init__()

        assert state_space is not None, "None state_space input: state_space should be selected."
        assert action_space is not None, "None action_space input: action_space should be selected."

        self.input = state_space
        self.lstm = nn.LSTM(input_size=state_space, hidden_size=hidden_size, num_layers=hidden_layers)
        self.Linear1 = nn.Linear(state_space + hidden_size + latent_dim, 32)
        self.Linear2 = nn.Linear(32, 32)
        self.Linear3 = nn.Linear(32, action_space)
        self.context_encoder = nn.Sequential(
            nn.Linear(context_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 2 * latent_dim)
        )
        self.latent_dim = latent_dim
        self.N_action = action_space
    
    def _product_of_gaussians(self, mus, sigmas_squared):
        '''
        compute mu, sigma of product of gaussians
        '''
        sigmas_squared = torch.clamp(sigmas_squared, min=1e-7)
        sigma_squared = 1. / torch.sum(torch.reciprocal(sigmas_squared), dim=0)
        mu = sigma_squared * torch.sum(mus / sigmas_squared, dim=0)
        return mu, sigma_squared

    def forward(self, x, hidden, context):
        # sample z
        params = self.context_encoder(context)
        mu = params[..., :self.latent_dim]
        sigma_squared = F.softplus(params[..., self.latent_dim:])
        z_params = [self._product_of_gaussians(m, s) for m, s in zip(torch.unbind(mu), torch.unbind(sigma_squared))]
        zmeans = torch.stack([p[0] for p in z_params])
        zvars = torch.stack([p[1] for p in z_params])
        posteriors = [torch.distributions.Normal(m, torch.sqrt(s)) for m, s in zip(torch.unbind(zmeans), torch.unbind(zvars))]
        z = torch.stack([d.rsample() for d in posteriors]).reshape(-1, 1, self.latent_dim)

        # forward
        h1, new_hidden = self.lstm(x, hidden)
        x = F.relu(self.Linear1(torch.cat((x, h1, z), dim=2)))
        x = F.relu(self.Linear2(x))
        return self.Linear3(x), new_hidden

    def sample_action(self, obs, hidden, context, epsilon):
        if random.random() < epsilon:
            a, h = self.forward(obs, hidden, context)
            return random.randint(0, self.N_action - 1), h
        else:
            a, h = self.forward(obs, hidden, context)
            return a.argmax().item(), h


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


def train(q_net=None, target_q_net=None, replay_buffer=None, device=None, optimizer=None, batch_size=64, learning_rate=2e-3, gamma=0.9, hidden=None, context=None):
    assert device is not None, "None Device input: device should be selected."

    # Get batch from replay buffer
    loss_func = nn.MSELoss()
    samples = replay_buffer.sample()

    h = (hidden[0].detach().clone(), hidden[1].detach().clone())

    states = torch.FloatTensor(samples["obs"]).to(device)
    actions = torch.LongTensor(samples["acts"].reshape(-1, 1)).to(device)
    rewards = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(device)
    next_states = torch.FloatTensor(samples["next_obs"]).to(device)
    dones = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(device)

    # Define loss
    q_target_max, _1 = target_q_net(next_states.reshape(next_states.shape[0], 1, next_states.shape[1]), h, context)
    targets = rewards + gamma * q_target_max.reshape(q_target_max.shape[0], q_target_max.shape[2]).max(1)[0].unsqueeze(1).detach() * dones
    q_out, _2 = q_net(states.reshape(states.shape[0], 1, states.shape[1]), h, context)
    q_a = q_out.reshape(q_out.shape[0], q_out.shape[2]).gather(1, actions)

    # Multiply Importance Sampling weights to loss
    loss = loss_func(q_a, targets)

    # Update Network
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def save_model(model, path='default.pth'):
    torch.save(model.state_dict(), path)

if __name__ == "__main__":
    # Set gym environment
    env = Environ()
    device = torch.device("cuda:0")
    batch_size = 32
    learning_rate = 2e-3
    buffer_len = int(20000)
    min_buffer_len = batch_size
    episodes = 4000
    print_per_iter = 20
    target_update_period = 1000
    steps = 0
    update_period = 10
    eps_start = 1.0
    eps_end = 0.05
    eps_decay = 0.997
    # tau = 1 * 1e-2
    pre_step = 200
    max_step = 200
    latent_dim = 5
    path = './DQN-context'
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
    hidden_layers = 2
    hidden_size = 128
    # read tasks
    with open('tasks.pkl', 'rb') as f:
        tasks = pickle.load(f)
    train_task = tasks[:200]
    test_task = tasks[100:]
    train_task_list = list(range(len(train_task)))
    test_task_list = list(range(len(test_task)))

    for i in range(env.n_ch):
        Q.append(Q_net(state_space=state_space, action_space=action_space, hidden_size=hidden_size, hidden_layers=hidden_layers, latent_dim=latent_dim, context_dim=env.get_context_dim()).to(device))
        Q_target.append(Q_net(state_space=state_space, action_space=action_space, hidden_size=hidden_size, hidden_layers=hidden_layers, latent_dim=latent_dim, context_dim=env.get_context_dim()).to(device))
        Q_target[i].load_state_dict(Q[i].state_dict())
        replay_buffer.append(ReplayBuffer(state_space, size=buffer_len, batch_size=batch_size))
        optimizer.append(optim.Adam(Q[i].parameters(), lr=learning_rate))

    epsilon = eps_start
    
    train_score_list = []
    test_score_list = []

    Trange = tqdm.tqdm(range(episodes))
    for i_step in Trange:
        # ------------------------------------------ set-training ------------------------------------------
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

        # ------------------------------------------ pre-sampling ------------------------------------------
        hidden_all = []
        for i in range(env.n_ch):
            hidden = (torch.zeros(hidden_layers, 1, hidden_size).float().to(device), torch.zeros(hidden_layers, 1, hidden_size).float().to(device))
            hidden_all.append(hidden)
        context_list = [[] for _ in range(env.n_ch)]
        for i in range(latent_dim):
            a = [0 for _ in range(env.n_ch)]
            for j in range(env.n_ch):
                a[j] = np.random.randint(0, env.action_dim)
            next_obs, r, done, _ = env.step(a)
            for j in range(env.n_ch):
                replay_buffer[j].put(obs[j], a[j], r.sum() / env.n_ch, next_obs[j], done)
                context_list[j].append(torch.concat((torch.tensor(obs[j]), torch.tensor(next_obs[j]), torch.tensor([r[j]]).to(torch.float32), env.decomposition_action_output(a[j]).to(torch.float32))))
            obs = next_obs
            e, j, s = env.reward_details()
            env.clear_reward()

        
        for t in range(pre_step):
            a = [0 for _ in range(env.n_ch)]
            for j in range(env.n_ch):
                a[j], hidden_all[j] = Q[j].sample_action(torch.from_numpy(obs[j]).reshape(-1, 1, Q[j].input).float().to(device), hidden_all[j], get_context(context_list[j]), epsilon)

            next_obs, r, done, _ = env.step(a)
            for j in range(env.n_ch):
                replay_buffer[j].put(obs[j], a[j], r.sum() / env.n_ch, next_obs[j], done)
                context_list[j].append(torch.concat((torch.tensor(obs[j]), torch.tensor(next_obs[j]), torch.tensor([r[j]]).to(torch.float32), env.decomposition_action_output(a[j]).to(torch.float32))))
            obs = next_obs
            e, j, s = env.reward_details()
            env.clear_reward()
        # ------------------------------------------ training ------------------------------------------
        for t in range(max_step):
            steps += 1
            a = [0 for _ in range(env.n_ch)]
            for j in range(env.n_ch):
                a[j], hidden_all[j] = Q[j].sample_action(torch.from_numpy(obs[j]).reshape(-1, 1, Q[j].input).float().to(device), hidden_all[j], get_context(context_list[j]), epsilon)

            next_obs, r, done, _ = env.step(a)
            for j in range(env.n_ch):
                replay_buffer[j].put(obs[j], a[j], r.sum() / env.n_ch, next_obs[j], done)
                context_list[j].append(torch.concat((torch.tensor(obs[j]), torch.tensor(next_obs[j]), torch.tensor([r[j]]).to(torch.float32), env.decomposition_action_output(a[j]).to(torch.float32))))
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
                        train(Q[j], Q_target[j], replay_buffer[j], device, optimizer=optimizer[j], batch_size=batch_size, learning_rate=learning_rate, hidden=hidden_all[j], context=get_context(context_list[j], batch_size))
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
        # ------------------------------------------ pre-sampling ------------------------------------------
        hidden_all = []
        for i in range(env.n_ch):
            hidden = (torch.zeros(hidden_layers, 1, hidden_size).float().to(device), torch.zeros(hidden_layers, 1, hidden_size).float().to(device))
            hidden_all.append(hidden)
        context_list = [[] for _ in range(env.n_ch)]
        for i in range(latent_dim):
            a = [0 for _ in range(env.n_ch)]
            for j in range(env.n_ch):
                a[j] = np.random.randint(0, env.action_dim)
            next_obs, r, done, _ = env.step(a)
            for j in range(env.n_ch):
                replay_buffer[j].put(obs[j], a[j], r.sum() / env.n_ch, next_obs[j], done)
                context_list[j].append(torch.concat((torch.tensor(obs[j]), torch.tensor(next_obs[j]), torch.tensor([r[j]]).to(torch.float32), env.decomposition_action_output(a[j]).to(torch.float32))))
            obs = next_obs
            e, j, s = env.reward_details()
            env.clear_reward()
        
        for t in range(pre_step):
            a = [0 for _ in range(env.n_ch)]
            for j in range(env.n_ch):
                a[j], hidden_all[j] = Q[j].sample_action(torch.from_numpy(obs[j]).reshape(-1, 1, Q[j].input).float().to(device), hidden_all[j], get_context(context_list[j]), epsilon)

            next_obs, r, done, _ = env.step(a)
            for j in range(env.n_ch):
                replay_buffer[j].put(obs[j], a[j], r.sum() / env.n_ch, next_obs[j], done)
                context_list[j].append(torch.concat((torch.tensor(obs[j]), torch.tensor(next_obs[j]), torch.tensor([r[j]]).to(torch.float32), env.decomposition_action_output(a[j]).to(torch.float32))))
            obs = next_obs
            e, j, s = env.reward_details()
            env.clear_reward()

        for t in range(max_step):
            a = [0 for _ in range(env.n_ch)]
            for j in range(env.n_ch):
                a[j], hidden_all[j] = Q[j].sample_action(torch.from_numpy(obs[j]).reshape(-1, 1, Q[j].input).float().to(device), hidden_all[j], get_context(context_list[j]), epsilon)
            next_obs, r, done, _ = env.step(a)
            for j in range(env.n_ch):
                replay_buffer[j].put(obs[j], a[j], r.sum() / env.n_ch, next_obs[j], done)
                context_list[j].append(torch.concat((torch.tensor(obs[j]), torch.tensor(next_obs[j]), torch.tensor([r[j]]).to(torch.float32), env.decomposition_action_output(a[j]).to(torch.float32))))
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
                        train(Q[j], Q_target[j], replay_buffer[j], device, optimizer=optimizer[j], batch_size=batch_size, learning_rate=learning_rate, hidden=hidden_all[j], context=get_context(context_list[j], batch_size))
            if done:
                break
        for i in range(env.n_ch):
            Q[i].load_state_dict(Q_params[i])

        # ------------------------------------------ saving ------------------------------------------
        Trange.set_postfix(train_score=train_score, test_score=test_score, epsilon=epsilon * 100)
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