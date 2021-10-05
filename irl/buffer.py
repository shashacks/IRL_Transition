import os
import numpy as np
import torch

class SerializedStartBuffer:
    def __init__(self, path, device):
        tmp = torch.load(path)
        self.buffer_size = self._n = tmp['time'].size(0)
        self.device = device

        self.times = tmp['time'].clone().to(self.device)
        self.qposs = tmp['qpos'].clone().to(self.device)
        self.qvels = tmp['qvel'].clone().to(self.device)
        self.qacts = tmp['qact'].clone().to(self.device)
        self.obstacle_poss = tmp['obstacle_pos'].clone().to(self.device)

    def sample(self):
        idx = np.random.randint(low=0, high=self._n)
        return (
            self.times[idx],
            self.qposs[idx],
            self.qvels[idx],
            self.qacts[idx],
            self.obstacle_poss[idx]
        )

class SerializedBufferForTSNE:

    def __init__(self, path, device):
        tmp = torch.load(path)
        self.buffer_size = self._n = tmp['state'].size(0)
        self.device = device

        self.states = tmp['state'].clone().to(self.device)
        self.actions = tmp['action'].clone().to(self.device)
        self.rewards = tmp['reward'].clone().to(self.device)
        self.dones = tmp['done'].clone().to(self.device)
        self.next_states = tmp['next_state'].clone().to(self.device)
        
    def sample(self, batch_size):
        idx = np.random.randint(low=0, high=self._n, size=batch_size)
        return (
            self.states[idx],
            self.actions[idx],
            self.rewards[idx],
            self.dones[idx],
            self.next_states[idx]
        )

class SerializedBuffer:

    def __init__(self, path, device):
        tmp = torch.load(path)
        self.buffer_size = self._n = tmp['state'].size(0)
        self.device = device

        self.states = tmp['state'].clone().to(self.device)
        self.actions = tmp['action'].clone().to(self.device)
        self.rewards = tmp['reward'].clone().to(self.device)
        self.dones = tmp['done'].clone().to(self.device)
        self.next_states = tmp['next_state'].clone().to(self.device)

        self.mean = torch.mean(self.states, dim=0)
        self.epsilon = torch.zeros(self.mean.shape) + torch.tensor([1e-4])
        self.std = torch.max(torch.std(self.states, dim=0), self.epsilon)

    def sample(self, batch_size):
        idx = np.random.randint(low=0, high=self._n, size=batch_size)
        return (
            self.states[idx],
            self.actions[idx],
            self.rewards[idx],
            self.dones[idx],
            self.next_states[idx]
        )

class StartBuffer:
    def __init__(self, buffer_size, time, qpos, qvel, qact, obstacle_pos, device):
        self._p = 0
        self.buffer_size = buffer_size
        self.device = device
        self.time = torch.empty(
            (buffer_size, time), dtype=torch.float, device=device)
        self.qpos = torch.empty(
            (buffer_size, qpos), dtype=torch.float, device=device)
        self.qvel = torch.empty(
            (buffer_size, qvel), dtype=torch.float, device=device)
        self.qact = torch.empty(
            (buffer_size, qact), dtype=torch.float, device=device)
        self.obstacle_pos = torch.empty(
            (buffer_size, obstacle_pos), dtype=torch.float, device=device)

    def append(self, time, qpos, qvel, qact, obstacle_pos):
        if not self._p == self.buffer_size:
            self.time[self._p].copy_(torch.from_numpy(time))
            self.qpos[self._p].copy_(torch.from_numpy(qpos))
            self.qvel[self._p].copy_(torch.from_numpy(qvel))
            self.qact[self._p].copy_(torch.from_numpy(qact))
            self.obstacle_pos[self._p].copy_(torch.from_numpy(obstacle_pos))

            self._p = self._p + 1

    def save(self, path):
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))

        torch.save({
            'time': self.time.clone().cpu(),
            'qpos': self.qpos.clone().cpu(),
            'qvel': self.qvel.clone().cpu(),
            'qact': self.qact.clone().cpu(),
            'obstacle_pos': self.obstacle_pos.clone().cpu()
        }, path)

    def sample(self):
        idx = np.random.randint(low=0, high=self.buffer_size)
        return (self.time[idx], self.qpos[idx], self.qvel[idx], self.qact[idx], self.obstacle_pos[idx])
        
class BufferForTSNE(SerializedBuffer):

    def __init__(self, buffer_size, state_shape, action_shape, device):
        self._n = 0
        self._p = 0
        self.buffer_size = buffer_size
        self.device = device

        self.states = torch.empty(
            (buffer_size, *state_shape), dtype=torch.float, device=device)
        self.actions = torch.empty(
            (buffer_size, *action_shape), dtype=torch.float, device=device)
        self.rewards = torch.empty(
            (buffer_size, 1), dtype=torch.float, device=device)
        self.dones = torch.empty(
            (buffer_size, 1), dtype=torch.float, device=device)
        self.next_states = torch.empty(
            (buffer_size, *state_shape), dtype=torch.float, device=device)

    def append(self, state, action, reward, done, next_state):
        self.states[self._p].copy_(torch.from_numpy(state))
        self.actions[self._p].copy_(torch.from_numpy(action))
        self.rewards[self._p] = float(reward)
        self.dones[self._p] = float(done)
        self.next_states[self._p].copy_(torch.from_numpy(next_state))

        self._p = (self._p + 1) % self.buffer_size
        self._n = min(self._n + 1, self.buffer_size)

    def save(self, path):
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))

        torch.save({
            'state': self.states.clone().cpu(),
            'action': self.actions.clone().cpu(),
            'reward': self.rewards.clone().cpu(),
            'done': self.dones.clone().cpu(),
            'next_state': self.next_states.clone().cpu()
        }, path)

class Buffer(SerializedBuffer):

    def __init__(self, buffer_size, state_shape, action_shape, device):
        self._n = 0
        self._p = 0
        self.buffer_size = buffer_size
        self.device = device

        self.states = torch.empty(
            (buffer_size, state_shape), dtype=torch.float, device=device)
        self.actions = torch.empty(
            (buffer_size, action_shape), dtype=torch.float, device=device)
        self.rewards = torch.empty(
            (buffer_size, 1), dtype=torch.float, device=device)
        self.dones = torch.empty(
            (buffer_size, 1), dtype=torch.float, device=device)
        self.next_states = torch.empty(
            (buffer_size, state_shape), dtype=torch.float, device=device)

    def append(self, state, action, reward, done, next_state):
        self.states[self._p].copy_(torch.from_numpy(state))
        self.actions[self._p].copy_(torch.from_numpy(action))
        self.rewards[self._p] = float(reward)
        self.dones[self._p] = float(done)
        self.next_states[self._p].copy_(torch.from_numpy(next_state))

        self._p = (self._p + 1) % self.buffer_size
        self._n = min(self._n + 1, self.buffer_size)

    def save(self, path):
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))

        torch.save({
            'state': self.states.clone().cpu(),
            'action': self.actions.clone().cpu(),
            'reward': self.rewards.clone().cpu(),
            'done': self.dones.clone().cpu(),
            'next_state': self.next_states.clone().cpu(),
        }, path)


class RolloutBuffer:

    def __init__(self, buffer_size, state_shape, action_shape, device, mix=1):
        self._n = 0
        self._p = 0
        self.mix = mix
        self.buffer_size = buffer_size
        self.total_size = mix * buffer_size

        self.states = torch.empty(
            (self.total_size, *state_shape), dtype=torch.float, device=device)
        self.actions = torch.empty(
            (self.total_size, *action_shape), dtype=torch.float, device=device)
        self.rewards = torch.empty(
            (self.total_size, 1), dtype=torch.float, device=device)
        self.dones = torch.empty(
            (self.total_size, 1), dtype=torch.float, device=device)
        self.log_pis = torch.empty(
            (self.total_size, 1), dtype=torch.float, device=device)
        self.next_states = torch.empty(
            (self.total_size, *state_shape), dtype=torch.float, device=device)

    def append(self, state, action, reward, done, log_pi, next_state):
        self.states[self._p].copy_(torch.from_numpy(state))
        self.actions[self._p].copy_(torch.from_numpy(action))
        self.rewards[self._p] = float(reward)
        self.dones[self._p] = float(done)
        self.log_pis[self._p] = float(log_pi)
        self.next_states[self._p].copy_(torch.from_numpy(next_state))

        self._p = (self._p + 1) % self.total_size
        self._n = min(self._n + 1, self.total_size)

    def get(self):
        assert self._p % self.buffer_size == 0
        start = (self._p - self.buffer_size) % self.total_size
        idx = slice(start, start + self.buffer_size)
        return (
            self.states[idx],
            self.actions[idx],
            self.rewards[idx],
            self.dones[idx],
            self.log_pis[idx],
            self.next_states[idx]
        )

    def sample(self, batch_size):
        assert self._p % self.buffer_size == 0
        idx = np.random.randint(low=0, high=self._n, size=batch_size)
        return (
            self.states[idx],
            self.actions[idx],
            self.rewards[idx],
            self.dones[idx],
            self.log_pis[idx],
            self.next_states[idx]
        )
