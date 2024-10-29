import os
import sys
from ENV.ENV_DDRQN import Environ
os.environ['PARL_BACKEND'] = 'torch'
import warnings

import parl
import gym
import numpy as np
from parl.utils import logger, summary
import argparse
import pandas as pd
import torch
import random
import pickle
import tqdm

from ppoalg_.uav_config import uav_config
from ppoalg_.env_utils import ParallelEnv, LocalEnv
from ppoalg_.uav_model import uavModel
from ppoalg_.agent import Agent
from ppoalg_.storage import RolloutStorage
from ppoalg_.multiPPO import PPO

def setup_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
setup_seed(520)

def run_evaluate_episodes(agent, eval_env, eval_episodes):
    eval_episode_rewards = []
    while len(eval_episode_rewards) < eval_episodes:
        obs = eval_env.reset()
        done = np.array([False] * eval_env.n_clusters)
        while not done.all():
            action = agent.predict(obs)
            obs, reward, done, info = eval_env.step(action)
        if "episode" in info.keys():
            eval_reward = info["episode"]["r"]
            eval_episode_rewards.append(eval_reward)
    return np.mean(eval_episode_rewards)

def main():
    # config
    config = uav_config # 离散动作空间
    if args.env_num:
        config['env_num'] = args.env_num
    config['env'] = args.env
    config['seed'] = args.seed
    config['test_every_steps'] = args.test_every_steps
    config['train_total_steps'] = args.train_total_steps
    config['batch_size'] = int(config['env_num'] * config['step_nums'])
    config['num_updates'] = int(config['train_total_steps'] // config['batch_size'])
    # env
    env = Environ()
    env = LocalEnv(env)

    eval_env = Environ()
    eval_env = LocalEnv(eval_env, test=True)

    obs_space = eval_env.obs_space
    act_space = eval_env.act_space
    n_clusters = eval_env.n_clusters
    # model
    model = uavModel(obs_space, act_space, n_clusters, args.device)
    ppo = PPO(
        model, args.device, clip_param=config['clip_param'], entropy_coef=config['entropy_coef'],
        initial_lr=config['initial_lr'], continuous_action=config['continuous_action']
    )
    agent = Agent(ppo, config, args.device)
    rollout = RolloutStorage(config['step_nums'], eval_env)
    # 忽略警告
    warnings.filterwarnings("ignore")

    # read tasks
    with open('tasks.pkl', 'rb') as f:
        tasks = pickle.load(f)
    train_task = tasks[:200]
    test_task = tasks[100:]
    train_task_list = list(range(len(train_task)))
    test_task_list = list(range(len(test_task)))

    # train
    done = np.zeros(env.n_clusters, dtype=np.float32)
    test_flag = 0
    total_steps = 0
    data = []
    pre_step = 200
    max_step = 200
    update_period = 10
    batch_size = 32
    path = './PPO'

    lmz = []
    energy = []
    hop = []
    suc = []
    train_score_list = []
    test_score_list = []

    Trange = tqdm.tqdm(range(1, 4001))

    for update in Trange:
        # -------------------------------- train --------------------------------
        # sample train tasks
        task_idx = np.random.choice(train_task_list)
        task = train_task[task_idx]
        obs = env.reset(task)
        train_score = 0
        a = []
        env.env.t_jammer = - env.env.jammer_start
        
        # reset replay buffer
        rollout.reset()
        done_template = np.array([False] * env.n_clusters)
        # sample and train
        for t in range(max_step):
            value, action, log_prob, _ = agent.sample(obs)
            next_obs, reward, next_done, info = env.step(action)
            rollout.append(obs, action, log_prob, reward, done_template, value)
            train_score += np.mean(reward)
            obs, done = next_obs, next_done
            env.env.clear_reward()
            if t % update_period == 0 and rollout.get_len() > batch_size:
                value = agent.value(obs)
                rollout.compute_returns(value, done_template)
                agent.learn(rollout, batch_size)
        
        # -------------------------------- test --------------------------------
        # sample train tasks
        params = model.get_params()
        task_idx = np.random.choice(test_task_list)
        task = test_task[task_idx]
        obs = env.reset(task)
        episode_energy = 0
        episode_jump = 0
        episode_suc = 0
        test_score = 0
        a = []
        env.env.t_jammer = - env.env.jammer_start
        
        # reset replay buffer
        rollout.reset()
        
        # sample and train
        for t in range(max_step):
            value, action, log_prob, _ = agent.sample(obs)
            next_obs, reward, next_done, info = env.step(action)
            rollout.append(obs, action, log_prob, reward, done_template, value)
            test_score += np.mean(reward)
            obs, done = next_obs, next_done
            e, j, s = env.env.reward_details()
            episode_energy += e
            episode_jump += j
            episode_suc += s
            env.env.clear_reward()
            if t % update_period == 0 and rollout.get_len() > batch_size:
                value = agent.value(obs)
                rollout.compute_returns(value, done_template)
                agent.learn(rollout, batch_size)
        model.load_params(params)
        
        lmz.append(train_score)
        energy.append(episode_energy)
        hop.append(episode_jump)
        suc.append(episode_suc)
        
        np.save(path + '/rew.npy', lmz)
        np.save(path + '/energy.npy', energy)
        np.save(path + '/hop.npy', hop)
        np.save(path + '/suc.npy', suc)

        train_score_list.append(train_score)
        test_score_list.append(test_score)

        DataFrame = pd.DataFrame([train_score_list, test_score_list], index = ['train', 'test']).T
        DataFrame.to_csv(path + '/reward.csv', index=False)
        Trange.set_postfix(train_score=train_score, test_score=test_score)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='uav-v0')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--env_num', type=int, default=None)
    parser.add_argument('--train_total_steps', type=int, default=int(1e7))
    parser.add_argument('--test_every_steps', type=int, default=int(5e3))
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()
    main()