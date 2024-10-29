import parl
import torch
import numpy as np
from parl.utils.scheduler import LinearDecayScheduler


class Agent(parl.Agent):
    def __init__(self, algorithm, config, device):
        super(Agent, self).__init__(algorithm)
        self.config = config
        self.device = device
        if self.config['lr_decay']:
            self.lr_scheduler = LinearDecayScheduler(self.config['initial_lr'], self.config['num_updates'])

    def predict(self, obs):
        obs = torch.tensor(obs)
        action = self.alg.predict(obs)
        return action.cpu().detach().numpy()
    
    def sample(self, obs):
        obs = torch.tensor(obs)
        value, action, action_log_probs, action_entropy = self.alg.sample(obs)
        value = np.array([value[i].cpu().detach().numpy() for i in range(self.alg.model.n_clusters)])
        action = action.cpu().detach().numpy() #[0]
        action_log_probs = action_log_probs.cpu().detach().numpy() #[0]
        action_entropy = action_entropy.cpu().detach().numpy()
        return value, action, action_log_probs, action_entropy
    
    def value(self, obs):
        obs = torch.tensor(obs)
        value = self.alg.value(obs)
        value = np.array([value[i].cpu().detach().numpy() for i in range(self.alg.model.n_clusters)])
        return value
    
    def learn(self, rollout, batch_size=32):
        # lr
        if self.config['lr_decay']:
            lr = self.lr_scheduler.step(step_num=1)
        else:
            lr = None

        sample_idx = np.random.choice(rollout.get_len(), batch_size)

        batch_obs, batch_action, batch_log_prob, batch_adv, batch_return, batch_value = rollout.sample_batch(sample_idx)

        batch_obs = torch.tensor(batch_obs).to(torch.device(self.device))
        batch_action = torch.tensor(batch_action).to(torch.device(self.device))
        batch_log_prob = torch.tensor(batch_log_prob).to(torch.device(self.device))
        batch_adv = torch.tensor(batch_adv).to(torch.device(self.device))
        batch_return = torch.tensor(batch_return).to(torch.device(self.device))
        batch_value = torch.tensor(batch_value).to(torch.device(self.device))

        value_loss, action_loss, entropy_loss = self.alg.learn(batch_obs, batch_action, batch_value, batch_return, batch_log_prob, batch_adv, lr)