import torch
from torch import nn

from .utils import build_mlp, reparameterize, evaluate_lop_pi

class StateIndependentPolicy(nn.Module):

    def __init__(self, state_shape, action_shape, hidden_units=(64, 64),
                 hidden_activation=nn.Tanh()):
        super().__init__()

        self.net = build_mlp(
            input_dim=state_shape[0],
            output_dim=action_shape[0],
            hidden_units=hidden_units,
            hidden_activation=hidden_activation
        )
        self.log_stds = nn.Parameter(torch.zeros(1, action_shape[0]))
        self.mean = torch.zeros(state_shape, dtype=torch.float32)
        self.std = torch.ones(state_shape, dtype=torch.float32)

    def forward(self, states):
        states = torch.clamp((states - self.mean) / self.std, min=-5.0, max=5.0)
        return torch.tanh(self.net(states))

    def sample(self, states):
        states = torch.clamp((states - self.mean) / self.std, min=-5.0, max=5.0)
        return reparameterize(self.net(states), self.log_stds)

    def evaluate_log_pi(self, states, actions):
        states = torch.clamp((states - self.mean) / self.std, min=-5.0, max=5.0)
        val = evaluate_lop_pi(self.net(states), self.log_stds, actions)
        # print(val)
        return val


class StateDependentPolicy(nn.Module):

    def __init__(self, state_shape, action_shape, hidden_units=(256, 256),
                 hidden_activation=nn.ReLU(inplace=True)):
        super().__init__()

        self.net = build_mlp(
            input_dim=state_shape[0],
            output_dim=2 * action_shape[0],
            hidden_units=hidden_units,
            hidden_activation=hidden_activation
        )
        self.mean = torch.zeros(state_shape, dtype=torch.float)
        self.std = torch.ones(state_shape, dtype=torch.float)

    def forward(self, states):
        states = torch.clamp((states - self.mean) / self.std, min=-5.0, max=5.0)
        return torch.tanh(self.net(states).chunk(2, dim=-1)[0])

    def sample(self, states):
        states = torch.clamp((states - self.mean) / self.std, min=-5.0, max=5.0)
        means, log_stds = self.net(states).chunk(2, dim=-1)
        return reparameterize(means, log_stds.clamp_(-20, 2))
