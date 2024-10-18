import numpy
from ENV.ENV_DDRQN import Environ
import math
import numpy as np
import random
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import pickle
import tqdm
import pandas as pd


def setup_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
setup_seed(520)

def get_decay(epi_iter):
    decay = math.pow(0.997, epi_iter)  # math.pow(a,b) a的b次方
    if decay < 0.05:
        decay = 0.05
    return decay

def get_context(context_list, batch=1):
    return torch.stack([torch.stack(random.sample(context_list, 5)) for _ in range(batch)])


# ------------------------ initialize ----------------------------


n_episode = 2000
n_steps = 200
n_pre_train = 200
latent_dim = 5
env = Environ()
n_episode_test = 100  # test episodes
steps = 0        #总步数
train_frequency = 10   # 10步训练一次
num_input = env.state_dim
num_output = env.action_dim
n_agent = env.n_ch
path = './DDQN-context'
ep_rewards = []
energy = []
hop = []
suc = []
train_score_list = []
test_score_list = []

with open('tasks.pkl', 'rb') as f:
    tasks = pickle.load(f)
train_task = tasks[:200]
test_task = tasks[100:]
train_task_list = list(range(len(train_task)))
test_task_list = list(range(len(test_task)))

if __name__ == '__main__':
    steps = 0
    Trange = tqdm.tqdm(range(n_episode))
    for i_episode in Trange:
        # ------------------------------------------ set-training ------------------------------------------
        task_idx = np.random.choice(train_task_list)
        task = train_task[task_idx]
        obs = env.reset(task)
        for i in range(env.n_ch):
            env.agents[i].buffer.reset()
        # ----------------- presample -----------------
        train_reward = 0

        hidden_all = []
        for i in range(n_agent):
            hidden = (Variable(torch.zeros(2, 1, 128).float()), Variable(torch.zeros(2, 1, 128).float()))
            hidden_all.append(hidden)
        context_list = [[] for _ in range(env.n_ch)]
        for i in range(latent_dim):
            action_all = [[] for _ in range(n_agent)]
            obs = env.get_state()
            for j in range(n_agent):
                action_all[j] = np.random.randint(0, env.action_dim)
            obs_, r, terminal, info = env.step(action_all)

            for j in range(n_agent):
                env.agents[j].remember(obs[j], action_all[j], r.sum() / n_agent, obs_[j])
                context_list[j].append(torch.concat((torch.tensor(obs[j]), torch.tensor(obs_[j]), torch.tensor([r[j]]).to(torch.float32), env.decomposition_action_output(action_all[j]).to(torch.float32))))
            env.clear_reward()

        for step in range(n_pre_train):
            action_all = [[] for _ in range(n_agent)]
            obs = env.get_state()

            for i in range(n_agent):
                action_all[i], hidden_all[i] = env.agents[i].get_action(obs[i], hidden_all[i], get_context(context_list[i]), get_decay(i_episode))
            obs_, r, terminal, info = env.step(action_all)
            for i in range(n_agent):
                env.agents[i].remember(obs[i], action_all[i], r.sum() / n_agent, obs_[i])
                context_list[i].append(torch.concat((torch.tensor(obs[i]), torch.tensor(obs_[i]), torch.tensor([r[i]]).to(torch.float32), env.decomposition_action_output(action_all[i]).to(torch.float32))))
            env.clear_reward()
        # ----------------- presample -----------------
        for step in range(n_steps):
            action_all = [[] for _ in range(n_agent)]
            obs = env.get_state()

            for i in range(n_agent):
                action_all[i], hidden_all[i] = env.agents[i].get_action(obs[i], hidden_all[i], get_context(context_list[i]), get_decay(i_episode))
            obs_, r, terminal, info = env.step(action_all)
            for i in range(n_agent):
                env.agents[i].remember(obs[i], action_all[i], r.sum() / n_agent, obs_[i])
            train_reward += (r.sum(axis=0) / n_agent)
            env.clear_reward()

            steps += 1
            if env.agents[0].buffer.is_available() and steps % train_frequency == 0:    # 随便检查一个的经验池即可
                for i in range(n_agent):
                    env.agents[i].train((hidden_all[i][0].detach().clone(), hidden_all[i][1].detach().clone()), get_context(context_list[i], 32))
        with open(path + '/params.pkl', 'wb') as f:
            pickle.dump(env.get_params(), f)
        # ------------------------------------------ set-testing ------------------------------------------
        params = env.get_params()
        task_idx = np.random.choice(test_task_list)
        task = test_task[task_idx]
        obs = env.reset(task)
        for i in range(env.n_ch):
            env.agents[i].buffer.reset()
        # ----------------- presample -----------------
        episode_reward = 0
        episode_energy = 0
        episode_jump = 0
        episode_suc = 0
        test_reward = 0

        hidden_all = []
        for i in range(n_agent):
            hidden = (Variable(torch.zeros(2, 1, 128).float()), Variable(torch.zeros(2, 1, 128).float()))
            hidden_all.append(hidden)
        context_list = [[] for _ in range(env.n_ch)]
        for i in range(latent_dim):
            action_all = [[] for _ in range(n_agent)]
            obs = env.get_state()
            for j in range(n_agent):
                action_all[j] = np.random.randint(0, env.action_dim)
            obs_, r, terminal, info = env.step(action_all)

            for j in range(n_agent):
                env.agents[j].remember(obs[j], action_all[j], r.sum() / n_agent, obs_[j])
                context_list[j].append(torch.concat((torch.tensor(obs[j]), torch.tensor(obs_[j]), torch.tensor([r[j]]).to(torch.float32), env.decomposition_action_output(action_all[j]).to(torch.float32))))
            env.clear_reward()

        for step in range(n_pre_train):
            action_all = [[] for _ in range(n_agent)]
            obs = env.get_state()

            for i in range(n_agent):
                action_all[i], hidden_all[i] = env.agents[i].get_action(obs[i], hidden_all[i], get_context(context_list[i]), get_decay(i_episode))
            obs_, r, terminal, info = env.step(action_all)
            for i in range(n_agent):
                env.agents[i].remember(obs[i], action_all[i], r.sum() / n_agent, obs_[i])
                context_list[i].append(torch.concat((torch.tensor(obs[i]), torch.tensor(obs_[i]), torch.tensor([r[i]]).to(torch.float32), env.decomposition_action_output(action_all[i]).to(torch.float32))))
            env.clear_reward()
        # ----------------- presample -----------------
        for step in range(n_steps):
            action_all = [[] for _ in range(n_agent)]
            obs = env.get_state()

            for i in range(n_agent):
                action_all[i], hidden_all[i] = env.agents[i].get_action(obs[i], hidden_all[i], get_context(context_list[i]), get_decay(i_episode))
            obs_, r, terminal, info = env.step(action_all)
            for i in range(n_agent):
                env.agents[i].remember(obs[i], action_all[i], r.sum() / n_agent, obs_[i])
                context_list[i].append(torch.concat((torch.tensor(obs[i]), torch.tensor(obs_[i]), torch.tensor([r[i]]).to(torch.float32), env.decomposition_action_output(action_all[i]).to(torch.float32))))
            test_reward += (r.sum(axis=0) / n_agent)
            e, j, s = env.reward_details()
            episode_energy += e
            episode_jump += j
            episode_suc += s
            env.clear_reward()

            steps += 1
            if env.agents[0].buffer.is_available() and steps % train_frequency == 0:    # 随便检查一个的经验池即可
                for i in range(n_agent):
                    env.agents[i].train((hidden_all[i][0].detach().clone(), hidden_all[i][1].detach().clone()), get_context(context_list[i], 32))
        env.load_params(params)

        Trange.set_postfix(train_score=train_reward, test_score=test_reward, epsilon=get_decay(i_episode) * 100)
        
        train_score_list.append(train_reward)
        test_score_list.append(test_reward)

        ep_rewards.append(episode_reward)
        energy.append(episode_energy)
        hop.append(episode_jump)
        suc.append(episode_suc)
        np.save(path + '/rew.npy', ep_rewards)
        np.save(path + '/energy.npy', energy)
        np.save(path + '/hop.npy', hop)
        np.save(path + '/suc.npy', suc)

        DataFrame = pd.DataFrame([train_score_list, test_score_list], index = ['train', 'test']).T
        DataFrame.to_csv(path + '/reward.csv', index=False)
