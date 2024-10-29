import parl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, Categorical
from parl.utils.utils import check_model_method
import numpy as np

__all__ = ['PPO']


class PPO(parl.Algorithm):
    def __init__(self, model, device, clip_param=0.1, value_loss_coef=0.5,
                entropy_coef=0.01, initial_lr=2.5e-4, eps=1e-5,  max_grad_norm=0.5,
                use_clipped_value_loss=True, norm_adv=True,
                continuous_action=False):
        """ PPO algorithm

        Args:
            model (parl.Model): forward network of actor and critic.
            clip_param (float): epsilon in clipping loss.
            value_loss_coef (float): value function loss coefficient in the optimization objective.
            entropy_coef (float): policy entropy coefficient in the optimization objective.
            initial_lr (float): learning rate.
            eps (float): Adam optimizer epsilon.
            max_grad_norm (float): max gradient norm for gradient clipping.
            use_clipped_value_loss (bool): whether or not to use a clipped loss for the value function.
            norm_adv (bool): whether or not to use advantages normalization.
            continuous_action (bool): whether or not is continuous action environment.
        """
        # check model method
        check_model_method(model, 'value', self.__class__.__name__)
        check_model_method(model, 'policy', self.__class__.__name__)

        assert isinstance(clip_param, float)
        assert isinstance(value_loss_coef, float)
        assert isinstance(entropy_coef, float)
        assert isinstance(initial_lr, float)
        assert isinstance(eps, float)
        assert isinstance(max_grad_norm, float)
        assert isinstance(use_clipped_value_loss, bool)
        assert isinstance(norm_adv, bool)
        assert isinstance(continuous_action, bool)

        self.clip_param = clip_param
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.norm_adv = norm_adv
        self.continuous_action = continuous_action

        self.device = device
        self.model = model.to(device)
        self.optimizer = [optim.Adam(self.model.net[i].parameters(), lr=initial_lr, eps=eps) for i in range(self.model.n_clusters)]
        # self.optimizer = [optim.SGD(self.model.net[i].parameters(), lr=initial_lr) for i in range(self.model.n_clusters)]

    def learn(self, batch_obs, batch_action, batch_value, batch_return, batch_logprob, batch_adv, lr=None):
        return_value_loss, return_action_loss, return_entropy_loss = [], [], []
        for i in range(self.model.n_clusters):
            values = self.model.net[i].value(batch_obs[i])
            logits = self.model.net[i].policy(batch_obs[i]) # shape(n_action, batch, n_act)

            #batch_action[i] is (batch, n_action)
            dist = [Categorical(logits=logits[j]) for j in range(len(logits))] # shape is (n_action, batch)    dist[][]就是有多少个动作, 每个动作有batch个分布
            action = batch_action[i].transpose(0, 1) # shape is (n_action, batch)
            action_log_probs = [dist[j].log_prob(action[j]) for j in range(len(logits))] # shape is (n_action, batch)
            dist_entropy = [dist[j].entropy() for j in range(len(logits))]
            entropy_loss = torch.stack(dist_entropy).mean()

            if self.norm_adv:
                batch_adv[i] = (batch_adv[i] - batch_adv[i].mean()) / (batch_adv[i].std() + 1e-8)
            
            ratio = torch.exp(torch.stack(action_log_probs) - batch_logprob[i].transpose(0, 1))
            surr1 = ratio * batch_adv[i]
            surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * batch_adv[i]
            action_loss = -torch.min(surr1, surr2).mean()

            values = values.view(-1)
            if self.use_clipped_value_loss:
                value_pred_clipped = batch_value[i] + torch.clamp(values - batch_value[i], -self.clip_param, self.clip_param)
                value_losses = (values - batch_return[i]).pow(2)
                value_losses_clipped = (value_pred_clipped - batch_return[i]).pow(2)
                value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = 0.5 * (batch_return[i] - values).pow(2).mean()
            loss = value_loss * self.value_loss_coef + action_loss - entropy_loss * self.entropy_coef
            
            if lr:
                for param_group in self.optimizer[i].param_groups:
                    param_group['lr'] = lr
            
            self.optimizer[i].zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.net[i].parameters(), self.max_grad_norm)
            self.optimizer[i].step()

            return_value_loss.append(value_loss.item())
            return_action_loss.append(action_loss.item())
            return_entropy_loss.append(entropy_loss.item())
        return np.mean(return_value_loss), np.mean(return_action_loss), np.mean(return_entropy_loss)

    def sample(self, obs):
        """ Define the sampling process. This function returns the action according to action distribution.
        
        Args:
            obs (torch tensor): observation, shape([batch_size] + obs_shape)
        Returns:
            value (torch tensor): value, shape([batch_size, 1])
            action (torch tensor): action, shape([batch_size] + action_shape)
            action_log_probs (torch tensor): action log probs, shape([batch_size])
            action_entropy (torch tensor): action entropy, shape([batch_size])
        """
        value = self.model.value(obs)
        
        logits = self.model.policy(obs)
        action = torch.zeros(size=(self.model.n_clusters, self.model.n_act), dtype=torch.int64, device=torch.device(self.device))
        action_log_probs = torch.zeros(size=(self.model.n_clusters, self.model.n_act), device=torch.device(self.device))
        action_entropy = torch.zeros(size=(self.model.n_clusters, self.model.n_act), device=torch.device(self.device))
        for i in range(self.model.n_clusters):
            for j in range(self.model.n_act):
                dist = Categorical(logits=logits[i][j])
                action[i][j] = dist.sample()
                action_log_probs[i][j] = dist.log_prob(action[i][j])
                action_entropy[i][j] = dist.entropy()

        return value, action, action_log_probs, action_entropy

    def predict(self, obs):
        """ use the model to predict action

        Args:
            obs (torch tensor): observation, shape([batch_size] + obs_shape)
        Returns:
            action (torch tensor): action, shape([batch_size] + action_shape),
                noted that in the discrete case we take the argmax along the last axis as action
        """
        logits = self.model.policy(obs)
        action = torch.zeros(size=(self.model.n_clusters, self.model.n_act), dtype=torch.int64, device=torch.device(self.model.device))
        for i in range(self.model.n_clusters):
            for j in range(self.model.n_act):
                dist = Categorical(logits=logits[i][j])
                action[i][j] = dist.sample()
        return action

    def value(self, obs):
        """ use the model to predict obs values

        Args:
            obs (torch tensor): observation, shape([batch_size] + obs_shape)
        Returns:
            value (torch tensor): value of obs, shape([batch_size])
        """
        return self.model.value(obs)
