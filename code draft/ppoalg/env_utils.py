import parl
import gym
import numpy as np
from parl.utils import logger

TEST_EPISODE = 3
ENV_DIM = 84
OBS_FORMAT = 'NCHW'
GAMMA = 0.99


class ParallelEnv(object):
    def __init__(self, config=None):
        self.config = config
        self.env_num = config['env_num']

        if config['seed']:
            self.env_list = [LocalEnv(config['env'], config['seed'] + i) for i in range(self.env_num)]
        else:
            self.env_list = [LocalEnv(config['env']) for _ in range(self.env_num)]
        
        if hasattr(self.env_list[0], '_max_episode_steps'):
            self._max_episode_steps = self.env_list[0]._max_episode_steps
        else:
            self._max_episode_steps = float('inf')

        self.total_steps = 0
        self.episode_steps_list = [0] * self.env_num
        self.episode_reward_list = [0] * self.env_num
        self.eval_ob_rms = None

    def reset(self):
        obs_list = []
        for env in self.env_list:
            obs = env.reset()
            obs_list.append(obs)
        self.obs_list = np.array(obs_list)
        return self.obs_list, None

    def step(self, action_list):
        next_obs_list, reward_list, done_list, info_list = [], [], [], []

        for i in range(self.env_num):
            self.total_steps += 1

            next_obs, reward, done, info = self.env_list[i].step(action_list[i])

            self.episode_steps_list[i] += 1
            self.episode_reward_list[i] += reward

            if done or self.episode_steps_list[i] >= self._max_episode_steps:
                next_obs = self.env_list[i].reset()
                self.episode_steps_list[i] = 0
                self.episode_reward_list[i] = 0
                if self.env_list[i].continuous_action:
                    # get running mean and variance of obs
                    self.eval_ob_rms = self.env_list[i].env.get_ob_rms()

            next_obs_list.append(next_obs)
            reward_list.append(reward)
            done_list.append(done)
            info_list.append(info)
        return np.array(next_obs_list), np.array(reward_list), np.array(done_list), np.array(info_list)


class LocalEnv(object):
    def __init__(self, env, env_seed=None, test=False, ob_rms=None):
        self.env = env

        # # is instance of gym.spaces.Box
        # if hasattr(env.action_space, 'high'):
        #     from parl.env.mujoco_wrappers import wrap_rms
        #     self._max_episode_steps = env._max_episode_steps
        #     self.continuous_action = True
        #     if test:
        #         self.env = wrap_rms(env, GAMMA, test=True, ob_rms=ob_rms)
        #     else:
        #         self.env = wrap_rms(env, gamma=GAMMA)
        # # is instance of gym.spaces.Discrete
        # elif hasattr(env.action_space, 'n'):
        #     from parl.env.atari_wrappers import wrap_deepmind
        #     self.continuous_action = False
        #     if test:
        #         self.env = wrap_deepmind(env, dim=ENV_DIM, obs_format=OBS_FORMAT, test=True, test_episodes=1)
        #     else:
        #         self.env = wrap_deepmind(env, dim=ENV_DIM, obs_format=OBS_FORMAT)
        # else:
        #     raise AssertionError('act_space must be instance of gym.spaces.Box or gym.spaces.Discrete')
        from parl.env.mujoco_wrappers import CompatWrapper, TimeLimitMaskEnv
        # self.env = CompatWrapper(self.env)
        # self.env = TimeLimitMaskEnv(self.env)
        # self.env = MonitorEnv(self.env)

        self.obs_space = (self.env.state_dim,)
        self.act_space = np.array((self.env.action_dim,))
        self.n_clusters = self.env.n_ch

    def reset(self, value):
        return self.env.reset(value)

    def step(self, action):
        return self.env.step(action)
    
class MonitorEnv(gym.Wrapper):
    """ Env wrapper that keeps tracks of total raw episode rewards, length of raw episode rewards for evaluation.
    """

    def __init__(self, env):
        super().__init__(env)
        self.rewards = None

    def step(self, action):
        ob, rew, done, info = self.env.step(action)
        self.update(ob, rew, done, info)
        return ob, rew, done, info

    def update(self, ob, rew, done, info):
        self.rewards.append(rew)
        if done.any():
            eprew = sum(self.rewards)
            eplen = len(self.rewards)
            epinfo = {
                "r": np.round(eprew, 6),
                "l": eplen,
            }
            assert isinstance(info, dict)
            info['episode'] = epinfo
            self.reset()

    def reset(self, **kwargs):
        self.rewards = []
        return self.env.reset(**kwargs)