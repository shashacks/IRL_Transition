import math
import torch
from torch import nn


def build_mlp(input_dim, output_dim, hidden_units=[64, 64],
              hidden_activation=nn.Tanh(), output_activation=None):
    layers = []
    units = input_dim
    for next_units in hidden_units:
        layers.append(nn.Linear(units, next_units))
        layers.append(hidden_activation)
        units = next_units
    layers.append(nn.Linear(units, output_dim))
    if output_activation is not None:
        layers.append(output_activation)
    return nn.Sequential(*layers)


def calculate_log_pi(log_stds, noises, actions):
    gaussian_log_probs = (-0.5 * noises.pow(2) - 0.5 * torch.log(2 * math.pi*log_stds.exp().pow(2))).sum(dim=-1, keepdim=True) 
    return gaussian_log_probs


def reparameterize(means, log_stds):
    actions = torch.normal(means, log_stds.exp())
    noises = (actions - means) / (log_stds.exp() + 1e-8)
    return actions, calculate_log_pi(log_stds, noises, actions)


def atanh(x):
    return 0.5 * (torch.log(1 + x + 1e-6) - torch.log(1 - x + 1e-6))


def evaluate_lop_pi(means, log_stds, actions):
    noises = (actions - means) / (log_stds.exp() + 1e-8)
    val = calculate_log_pi(log_stds, noises, actions)
    # print(val)
    # exit()
    return val
