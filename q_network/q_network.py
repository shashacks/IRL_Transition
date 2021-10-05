import os
import os.path as osp

import torch
from torch import nn
from torch.optim import Adam
import random
from q_network.replay_buffer import ReplayBuffer

class DQN(nn.Module):
    def __init__(self, num_inputs, actions_dim):
        super(DQN, self).__init__()

        self.nn = nn.Sequential(
            nn.Linear(num_inputs, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, actions_dim)
        )

        self.mean = torch.zeros(num_inputs, dtype=torch.float32)
        self.std = torch.ones(num_inputs, dtype=torch.float32)

    def forward(self, x):
        x = torch.clamp((x - self.mean) / self.std, min=-5.0, max=5.0)
        return self.nn(x)

class CnnDQN(nn.Module):
    def __init__(self, inputs_shape, num_actions):
        super(CnnDQN, self).__init__()

        self.inut_shape = inputs_shape
        self.num_actions = num_actions

        self.features = nn.Sequential(
            nn.Conv2d(inputs_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(self.features_size(), 512),
            nn.ReLU(),
            nn.Linear(512, self.num_actions)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def features_size(self):
        return self.features(torch.zeros(1, *self.inut_shape)).view(1, -1).size(1)

class DQN_Converter:
    def __init__(self, args, state_dim, batch_size):
        self.args = args
        self.is_training = True
        self.batch_size = batch_size
        self.buffer = ReplayBuffer(self.args.max_buffer_size_q)
        

        self.model = DQN(state_dim, 2)
        self.target_model = DQN(state_dim, 2)
        self.target_model.load_state_dict(self.model.state_dict())
        self.model_optim = Adam(self.model.parameters(), lr=self.args.learning_rate_q)

    def act(self, state):
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0)
        q_value = self.model.forward(state)
        action = q_value.max(1)[1].item()
        return action

    def learning(self, fr):
        s0, a, r, s1, done = self.buffer.sample(self.batch_size)

        s0 = torch.tensor(s0, dtype=torch.float)
        s1 = torch.tensor(s1, dtype=torch.float)
        a = torch.tensor(a, dtype=torch.long)
        r = torch.tensor(r, dtype=torch.float)
        done = torch.tensor(done, dtype=torch.float)

        q_values = self.model(s0)
        next_q_values = self.model(s1)
        next_q_state_values = self.target_model(s1)
        q_value = q_values.gather(1, a.unsqueeze(1)).squeeze(1)

        next_q_value = next_q_state_values.gather(1, next_q_values.max(1)[1].unsqueeze(1)).squeeze(1)
        expected_q_value = r + self.args.gamma_q * next_q_value * (1 - done)
        loss = (q_value - expected_q_value.detach()).pow(2).mean()

        self.model_optim.zero_grad()
        loss.backward()
        self.model_optim.step()

        if fr % self.args.update_target_frequency_q == 0:
            self.target_model.load_state_dict(self.model.state_dict())

        return loss.item()

    def save_model(self, fpath, fname):
        print(fpath, fname)
        fname = osp.join(fpath, fname)
        os.makedirs(fpath, exist_ok=True)
        torch.save({
            'mean': self.model.mean,
            'std': self.model.std,
            'model': self.model.state_dict(),
            }, fname)
    
    def load_weights(self, fanme=None):
        checkpoint = torch.load(fanme)
        self.model.mean = checkpoint['mean']
        self.model.std = checkpoint['std']
        self.model.load_state_dict(checkpoint['model'])
        
    
    def set_mean_std(self, mean, std):
        self.model.mean = mean
        self.model.std = std
        self.target_model.mean = mean
        self.target_model.std = std
