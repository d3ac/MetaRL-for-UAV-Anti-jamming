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
    decay = math.pow(0.98, epi_iter)  # math.pow(a,b) a的b次方
    if decay < 0.05:
        decay = 0.05
    return decay

def get_context(context_list, batch=1, device=torch.device("cuda:0")):
    return torch.stack([torch.stack(random.sample(context_list, 5)) for _ in range(batch)]).to(device)



# ------------------------ initialize ----------------------------

device = torch.device("cuda:0")
episodes = 1000
n_steps = 100
batch_size = 16
latent_dim = 64
n_tasks = 30
env = Environ(device=device, test=True)
steps = 0        #总步数
train_frequency = 40   # 10步训练一次
num_input = env.state_dim
num_output = env.action_dim
n_agent = env.n_ch
path_pre = './DDQN-context'
path = './DDQN-context-test'
ep_rewards = []
energy = []
hop = []
suc = []
train_score_list = []
test_score_list = []
for i in range(episodes):
    train_score_list.append(0)
    test_score_list.append(0)
    ep_rewards.append(0)
    energy.append(0)
    hop.append(0)
    suc.append(0)
train_score_list = np.array(train_score_list)
test_score_list = np.array(test_score_list)
ep_rewards = np.array(ep_rewards)
energy = np.array(energy)
hop = np.array(hop)
suc = np.array(suc)

with open('tasks.pkl', 'rb') as f:
    tasks = pickle.load(f)
train_task = tasks[:200]
test_task = tasks[100:]
train_task_list = list(range(len(train_task)))
test_task_list = list(range(len(test_task)))

if __name__ == '__main__':
    steps = 0
    TR = tqdm.tqdm(range(n_tasks))
    for N_task in TR:
        Trange = tqdm.tqdm(range(episodes))
        with open(path_pre + '/params.pkl', 'rb') as f:
            params = pickle.load(f)
        env.load_params(params)
        task_idx = np.random.choice(test_task_list)
        task = test_task[task_idx]
        for i in range(env.n_ch):
            env.agents[i].buffer.reset()
        hidden_all = []
        for i in range(n_agent):
            hidden = (torch.zeros(2, 1, 128).float().to(device), torch.zeros(2, 1, 128).float().to(device))
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

        for i_episode in Trange:
            # ------------------------------------------ set-testing ------------------------------------------
            obs = env.reset(task)
            # ----------------- presample -----------------
            episode_reward = 0
            episode_energy = 0
            episode_jump = 0
            episode_suc = 0
            test_reward = 0

            # ----------------- presample -----------------
            for step in range(n_steps):
                action_all = [[] for _ in range(n_agent)]
                obs = env.get_state()

                for i in range(n_agent):
                    action_all[i], hidden_all[i] = env.agents[i].get_action(obs[i], hidden_all[i], get_context(context_list[i]), get_decay(i_episode), device=device)
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
                        env.agents[i].train((hidden_all[i][0].detach().clone(), hidden_all[i][1].detach().clone()), get_context(context_list[i], batch_size), batch_size=batch_size, device=device)

            Trange.set_postfix(train_score=0, test_score=test_reward, epsilon=get_decay(i_episode) * 100)
            
            # train_score_list.append(train_reward)
            train_score_list[i_episode] += 0
            # test_score_list.append(test_reward)
            test_score_list[i_episode] += test_reward
            # ep_rewards.append(episode_reward)
            ep_rewards[i_episode] += 0
            # energy.append(episode_energy)
            energy[i_episode] += episode_energy
            # hop.append(episode_jump)
            hop[i_episode] += episode_jump
            # suc.append(episode_suc)
            suc[i_episode] += episode_suc

            np.save(path + '/rew.npy', ep_rewards/(N_task+1))
            np.save(path + '/energy.npy', energy/(N_task+1))
            np.save(path + '/hop.npy', hop/(N_task+1))
            np.save(path + '/suc.npy', suc/(N_task+1))

            DataFrame = pd.DataFrame([train_score_list, test_score_list/(N_task+1)], index = ['train', 'test']).T
            DataFrame.to_csv(path + '/reward.csv', index=False)
