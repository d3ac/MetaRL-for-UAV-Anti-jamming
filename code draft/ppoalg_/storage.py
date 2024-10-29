import numpy as np


class RolloutStorage():
    def __init__(self, step_nums, env):
        self.obs = [np.zeros((step_nums,) + (env.obs_space[0],), dtype='float32') for i in range(env.n_clusters)]
        self.actions = [np.zeros((step_nums,) + (len(env.act_space),), dtype='float32') for i in range(env.n_clusters)]
        self.logprobs = [np.zeros((step_nums,) + (len(env.act_space),), dtype='float32') for i in range(env.n_clusters)]
        self.rewards = [np.zeros((step_nums,), dtype='float32') for i in range(env.n_clusters)]
        self.dones = [np.zeros((step_nums,), dtype='float32') for i in range(env.n_clusters)]
        self.values = [np.zeros((step_nums,), dtype='float32') for i in range(env.n_clusters)]

        self.step_nums = step_nums
        self.obs_space = (env.obs_space[0],)
        self.act_space = (len(env.act_space),)
        self.n_clusters = env.n_clusters
        self.cur_step = np.zeros(env.n_clusters, dtype='int32')
    
    def get_len(self):
        return self.cur_step[0]
    
    def reset(self):
        for i in range(self.n_clusters):
            self.cur_step[i] = 0
    
    def append(self, obs, action, logprob, reward, done, value):
        for i in range(self.n_clusters):
            self.obs[i][self.cur_step[i]] = obs[i]
            self.actions[i][self.cur_step[i]] = action[i]
            self.logprobs[i][self.cur_step[i]] = logprob[i]
            self.rewards[i][self.cur_step[i]] = reward[i]
            self.dones[i][self.cur_step[i]] = done[i]
            self.values[i][self.cur_step[i]] = value[i]
            self.cur_step[i] = (self.cur_step[i] + 1) % self.step_nums

    def compute_returns(self, value, done, gamma=0.99, gae_lambda=0.95):
        advantages = [np.zeros_like(self.rewards[i]) for i in range(self.n_clusters)]
        returns = []
        for i in range(self.n_clusters):
            last_gae_lambda = 0
            for t in reversed(range(self.step_nums)):
                if t == self.step_nums - 1:
                    next_non_terminal = 1.0 - done[i]
                    next_values = value[i].reshape(1, -1)
                else:
                    next_non_terminal = 1.0 - self.dones[i][t+1]
                    next_values = self.values[i][t+1]
                delta = self.rewards[i][t] + gamma * next_values * next_non_terminal - self.values[i][t]
                advantages[i][t] = last_gae_lambda = delta + gamma * gae_lambda * next_non_terminal * last_gae_lambda
            returns.append(advantages[i] + self.values[i])
        self.advantages = advantages
        self.returns = returns
        return advantages, returns
    
    def sample_batch(self, idx):
        b_obs = [self.obs[i].reshape((-1,)+self.obs_space) for i in range(self.n_clusters)]
        b_logprobs = [self.logprobs[i].reshape((-1,)+self.act_space) for i in range(self.n_clusters)]
        b_actions = [self.actions[i].reshape((-1,)+self.act_space) for i in range(self.n_clusters)]
        b_advantages = [self.advantages[i].reshape((-1,)) for i in range(self.n_clusters)]
        b_returns = [self.returns[i].reshape((-1,)) for i in range(self.n_clusters)]
        b_values = [self.values[i].reshape((-1,)) for i in range(self.n_clusters)]

        obs = np.array([b_obs[i][idx] for i in range(self.n_clusters)])
        logprobs = np.array([b_logprobs[i][idx] for i in range(self.n_clusters)])
        actions = np.array([b_actions[i][idx] for i in range(self.n_clusters)])
        advantages = np.array([b_advantages[i][idx] for i in range(self.n_clusters)])
        returns = np.array([b_returns[i][idx] for i in range(self.n_clusters)])
        values = np.array([b_values[i][idx] for i in range(self.n_clusters)])

        return obs, actions, logprobs, advantages, returns, values