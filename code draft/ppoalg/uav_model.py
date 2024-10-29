import torch
import parl
from torch import nn
import torch.nn.functional as F

class baseModel(nn.Module):
    def __init__(self, obs_shape, act_shape, device):
        super(baseModel, self).__init__()
        self.device = device
        self.fc1 = nn.Linear(obs_shape[0], 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc_pi = nn.ModuleList([nn.Linear(32, act_shape[i]).to(torch.device(self.device)) for i in range(len(act_shape))])
        self.fc_v = nn.Linear(32, 1)
        # init weight
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def value(self, obs):
        obs = obs.to(torch.device(self.device)).to(torch.float32)
        obs = F.relu(self.fc1(obs))
        obs = F.relu(self.fc2(obs))
        v = self.fc_v(obs)
        return v.reshape(-1)
    
    def policy(self, obs): # 注意返回的是 (n_action, batch_size, n_act)
        obs = obs.to(torch.device(self.device)).to(torch.float32)
        obs = F.relu(self.fc1(obs))
        obs = F.relu(self.fc2(obs))
        logits = [self.fc_pi[i](obs) for i in range(len(self.fc_pi))]
        return logits

class uavModel(parl.Model):
    def __init__(self, obs_space, act_space, n_clusters, device):
        """
        obs_space: (obs_n,)
        act_space: (n, n, n, n, ...)
        """
        super(uavModel, self).__init__()
        self.net = [baseModel(obs_space, act_space, device) for i in range(n_clusters)]
        for i in range(n_clusters):
            self.net[i].to(torch.device(device))
        self.n_clusters = n_clusters
        self.n_act = len(act_space)
        self.device = device
    
    # 如果是调用下面两个, 那应该是 (n_clusters, xx) 的输入, xx 还需要batch一下
    def value(self, obs):
        return [self.net[i].value(obs[i].reshape(1, -1)) for i in range(len(self.net))]
    
    def policy(self, obs):
        return [self.net[i].policy(obs[i].reshape(1, -1)) for i in range(len(self.net))]
    
    def get_params(self):
        return [self.net[i].state_dict() for i in range(len(self.net))]
    
    def load_params(self, params):
        for i in range(len(self.net)):
            self.net[i].load_state_dict(params[i])


# --------------------------------------------------------- context ---------------------------------------------------------
class baseModel_context(nn.Module):
    def __init__(self, obs_shape, act_shape, device):
        super(baseModel_context, self).__init__()
        self.device = device
        self.lstm = nn.LSTM(input_size = obs_shape[0], hidden_size = 128, num_layers = 2)
        self.fc1 = nn.Linear(obs_shape[0] + 128, 32)
        self.fc1_policy = nn.Linear(obs_shape[0], 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc_pi = nn.ModuleList([nn.Linear(32, act_shape[i]).to(torch.device(self.device)) for i in range(len(act_shape))])
        self.fc_v = nn.Linear(32, 1)
        # init weight
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def value(self, obs, hidden):
        obs = obs.to(torch.device(self.device)).to(torch.float32).reshape(obs.shape[0], 1, -1)
        x, new_hidden = self.lstm(obs, hidden)
        obs = F.relu(self.fc1(torch.cat((x, obs), dim=2)))
        obs = F.relu(self.fc2(obs))
        v = self.fc_v(obs)
        return v.reshape(-1), new_hidden
    
    def policy(self, obs): # 注意返回的是 (n_action, batch_size, n_act)
        obs = obs.to(torch.device(self.device)).to(torch.float32)
        obs = F.relu(self.fc1_policy(obs))
        obs = F.relu(self.fc2(obs))
        logits = [self.fc_pi[i](obs) for i in range(len(self.fc_pi))]
        return logits

class uavModel_context(parl.Model):
    def __init__(self, obs_space, act_space, n_clusters, device):
        """
        obs_space: (obs_n,)
        act_space: (n, n, n, n, ...)
        """
        super(uavModel_context, self).__init__()
        self.net = [baseModel_context(obs_space, act_space, device) for i in range(n_clusters)]
        for i in range(n_clusters):
            self.net[i].to(torch.device(device))
        self.n_clusters = n_clusters
        self.n_act = len(act_space)
        self.device = device
    
    def value(self, obs, hidden):
        values = []
        hiddens = []
        for i in range(len(self.net)):
            value, hidden_ = self.net[i].value(obs[i].reshape(1, -1), hidden[i])
            values.append(value)
            hiddens.append(hidden_)
        return values, hiddens
    
    def policy(self, obs):
        policies = []
        for i in range(len(self.net)):
            policy = self.net[i].policy(obs[i].reshape(1, -1))
            policies.append(policy)
        return policies
    
    def get_params(self):
        return [self.net[i].state_dict() for i in range(len(self.net))]
    
    def load_params(self, params):
        for i in range(len(self.net)):
            self.net[i].load_state_dict(params[i])