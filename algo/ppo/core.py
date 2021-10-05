import numpy as np
import scipy.signal
from gym.spaces import Box, Discrete
from mpi4py import MPI
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input: 
        vector x, 
        [x0, 
         x1, 
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,  
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class Actor(nn.Module):

    def _distribution(self, obs):
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError

    def forward(self, obs, act=None):
        # Produce action distributions for given observations, and 
        # optionally compute the log likelihood of given actions under
        # those distributions.
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a


class MLPCategoricalActor(Actor):
    
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.logits_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        logits = self.logits_net(obs)
        return Categorical(logits=logits)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)


class MLPGaussianActor(Actor):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, rms=True):
        super().__init__()
        self.obs_dim = obs_dim
        log_std = -0.08 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self.mu_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)
        self.pi = torch.tensor(np.pi, dtype=torch.float32)
        self.e = torch.tensor(np.e, dtype=torch.float32)
        self.rms = rms

        with torch.no_grad():
            self.rms_sum = torch.zeros(obs_dim, dtype=torch.float32) 
            self.rms_sumsq = torch.tensor(np.ones(obs_dim) * 1e-2, dtype=torch.float32)
            self.rms_count = torch.tensor(1e-2, dtype=torch.float32)
            self.rms_eps = torch.tensor(1e-2, dtype=torch.float32)
            
            self.rms_mean = self.rms_sum / self.rms_count
            self.rms_std = torch.sqrt(torch.max((self.rms_sumsq / self.rms_count) - torch.pow(self.rms_mean, 2), self.rms_eps))

    def _distribution(self, obs):
        obs = torch.clamp((obs - self.rms_mean) / self.rms_std, min=-5.0, max=5.0)
        mu = self.mu_net(obs)
        std = torch.exp(self.log_std)
        return Normal(mu, std)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(axis=-1)    # Last axis sum needed for Torch Normal distribution
    
    def entropy(self):
        return torch.sum(self.log_std + .5*torch.log(2.0 * self.pi * self.e))

    def update(self, x, n):
        self.rms_sum = self.rms_sum +  torch.tensor(x[0:n], dtype=torch.float32)
        self.rms_sumsq = self.rms_sumsq +  torch.tensor(x[n:2*n], dtype=torch.float32)
        self.rms_count = self.rms_count + torch.tensor(x[2*n], dtype=torch.float32)

        self.rms_mean =  self.rms_sum / self.rms_count
        self.rms_std = torch.sqrt(torch.max((self.rms_sumsq / self.rms_count) - torch.pow(self.rms_mean, 2), self.rms_eps))
        if self.rms == False:
            self.rms_mean = torch.zeros(self.obs_dim, dtype=torch.float32)
            self.rms_std = torch.ones(self.obs_dim, dtype=torch.float32)

class MLPCritic(nn.Module):

    def __init__(self, obs_dim, hidden_sizes, activation, rms=True):
        super().__init__()
        self.obs_dim = obs_dim
        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)
        self.rms = rms
        with torch.no_grad():
            self.rms_sum = torch.zeros(obs_dim, dtype=torch.float32) 
            self.rms_sumsq = torch.tensor(np.ones(obs_dim) * 1e-2, dtype=torch.float32)
            self.rms_count = torch.tensor(1e-2, dtype=torch.float32)
            self.rms_eps = torch.tensor(1e-2, dtype=torch.float32)

            self.rms_mean = self.rms_sum / self.rms_count
            self.rms_std = torch.sqrt(torch.max((self.rms_sumsq / self.rms_count) - torch.pow(self.rms_mean, 2), self.rms_eps))


    def forward(self, obs):
        obs = torch.clamp((obs - self.rms_mean) / self.rms_std, min=-5.0, max=5.0)
        return torch.squeeze(self.v_net(obs), -1) # Critical to ensure v has right shape.

    def update(self, x, n):
        self.rms_sum = self.rms_sum +  torch.tensor(x[0:n], dtype=torch.float32)
        self.rms_sumsq = self.rms_sumsq +  torch.tensor(x[n:2*n], dtype=torch.float32)
        self.rms_count = self.rms_count + torch.tensor(x[2*n], dtype=torch.float32)

        self.rms_mean =  self.rms_sum / self.rms_count
        self.rms_std = torch.sqrt(torch.max((self.rms_sumsq / self.rms_count) - torch.pow(self.rms_mean, 2), self.rms_eps))
        if self.rms == False:
            self.rms_mean = torch.zeros(self.obs_dim, dtype=torch.float32)
            self.rms_std = torch.ones(self.obs_dim, dtype=torch.float32)

class MLPActorCritic(nn.Module):
    def __init__(self, observation_space, action_space, 
                 hidden_sizes=(64,64), activation=nn.Tanh, rms=True):
        super().__init__()

        obs_dim = observation_space.shape[0]
        self.obs_dim = obs_dim

        # policy builder depends on action space
        if isinstance(action_space, Box):
            self.pi = MLPGaussianActor(obs_dim, action_space.shape[0], hidden_sizes, activation, rms)
        elif isinstance(action_space, Discrete):
            self.pi = MLPCategoricalActor(obs_dim, action_space.n, hidden_sizes, activation, rms)

        # build value function
        self.v  = MLPCritic(obs_dim, hidden_sizes, activation)

    def step(self, obs):
        with torch.no_grad():
            pi = self.pi._distribution(obs)
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v(obs)
        return a.numpy(), v.numpy(), logp_a.numpy()

    def act(self, obs):
        return self.step(obs)[0]

    def update(self, x):
        with torch.no_grad():
            x = x.numpy().astype('float64')
            n = int(self.obs_dim)
            totalvec = np.zeros(n*2+1, dtype=np.float64)
            addvec = np.concatenate([x.sum(axis=0).ravel(), np.square(x).sum(axis=0).ravel(), np.array([len(x)],dtype=np.float32)])

            MPI.COMM_WORLD.Allreduce(addvec, totalvec, op=MPI.SUM)
            self.pi.update(totalvec, n)
            self.v.update(totalvec, n)