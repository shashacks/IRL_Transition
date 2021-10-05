import os
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam

from irl.algo.ppo import PPO
from irl.network.disc import AIRLDiscrim


class AIRL(PPO):

    def __init__(self, args, buffer_exp, start_buffer_exp, state_shape, action_shape, device, seed, front,
                 gamma=0.995, rollout_length=10000, mix_buffer=1,
                 batch_size=64, lr_actor=3e-4, lr_critic=3e-4, lr_disc=3e-4,
                 units_actor=(64, 64), units_critic=(64, 64),
                 units_disc_r=(100, 100), units_disc_v=(100, 100),
                 epoch_ppo=50, epoch_disc=10, clip_eps=0.2, lambd=0.97,
                 coef_ent=0.0, max_grad_norm=10.0):
        super().__init__(
            state_shape, action_shape, device, seed, start_buffer_exp, gamma, rollout_length,
            mix_buffer, lr_actor, lr_critic, units_actor, units_critic,
            epoch_ppo, clip_eps, lambd, coef_ent, max_grad_norm
        )
        self.front = front
        self.args = args

        # Expert's buffer.
        self.buffer_exp = buffer_exp


        # Discriminator.
        self.disc = AIRLDiscrim(
            state_shape=state_shape,
            action_shape=action_shape,
            gamma=gamma,
            hidden_units_r=units_disc_r,
            hidden_units_v=units_disc_v,
            hidden_activation_r=nn.ReLU(inplace=True),
            hidden_activation_v=nn.ReLU(inplace=True)
        ).to(device)

        self.learning_steps_disc = 0
        self.optim_disc = Adam(self.disc.parameters(), lr=lr_disc)
        self.batch_size = batch_size
        self.epoch_disc = epoch_disc

        self.actor.mean = self.buffer_exp.mean
        self.actor.std = self.buffer_exp.std
        self.critic.mean = self.buffer_exp.mean
        self.critic.std = self.buffer_exp.std

    def update(self, writer):
        self.learning_steps += 1

        for _ in range(self.epoch_disc):
            self.learning_steps_disc += 1

            # Samples from current policy's trajectories.
            states, actions, _, dones, log_pis, next_states = \
                self.buffer.sample(self.batch_size)
            # Samples from expert's demonstrations.
            states_exp, actions_exp, _, dones_exp, next_states_exp = \
                self.buffer_exp.sample(self.batch_size)
            # Calculate log probabilities of expert actions.
            with torch.no_grad():
                log_pis_exp = self.actor.evaluate_log_pi(states_exp, actions_exp)
 
            n_states = torch.clamp((states - self.buffer_exp.mean) / self.buffer_exp.std, min=-5.0, max=5.0)
            n_next_states = torch.clamp((next_states - self.buffer_exp.mean) / self.buffer_exp.std, min=-5.0, max=5.0)
            n_states_exp = torch.clamp((states_exp - self.buffer_exp.mean) / self.buffer_exp.std, min=-5.0, max=5.0)
            n_next_states_exp = torch.clamp((next_states_exp - self.buffer_exp.mean) / self.buffer_exp.std, min=-5.0, max=5.0)

            # Update discriminator.
            self.update_disc(
                n_states, actions, dones, log_pis, n_next_states, 
                n_states_exp, actions_exp, dones_exp, log_pis_exp, n_next_states_exp, writer
            )

        # We don't use reward signals here,
        states, actions, _, dones, log_pis, next_states = self.buffer.get()
        
        n_states = torch.clamp((states - self.buffer_exp.mean) / self.buffer_exp.std, min=-5.0, max=5.0)
        n_next_states = torch.clamp((next_states - self.buffer_exp.mean) / self.buffer_exp.std, min=-5.0, max=5.0)

        # Calculate rewards.
        rewards = self.disc.calculate_reward(
            n_states, actions, dones, log_pis, n_next_states)

        # Update PPO using estimated rewards.
        self.update_ppo(states, actions, rewards, dones, log_pis, next_states, writer)

    def update_disc(self, states, actions, dones, log_pis, next_states,
                    states_exp, action_exp, dones_exp, log_pis_exp,
                    next_states_exp, writer):
        # Output of discriminator is (-inf, inf), not [0, 1].
        logits_pi = self.disc(states, actions, dones, log_pis, next_states)
        logits_exp = self.disc(states_exp, action_exp, dones_exp, log_pis_exp, next_states_exp)

        # Discriminator is to maximize E_{\pi} [log(1 - D)] + E_{exp} [log(D)].
        loss_pi = -F.logsigmoid(-logits_pi).mean()
        loss_exp = -F.logsigmoid(logits_exp).mean()
        loss_disc = loss_pi + loss_exp

        self.optim_disc.zero_grad()
        loss_disc.backward()
        self.optim_disc.step()

        if self.learning_steps_disc % self.epoch_disc == 0:
            writer.add_scalar(
                'loss/disc', loss_disc.item(), self.learning_steps)

            # Discriminator's accuracies.
            with torch.no_grad():
                acc_pi = (logits_pi < 0).float().mean().item()
                acc_exp = (logits_exp > 0).float().mean().item()
            writer.add_scalar('stats/acc_pi', acc_pi, self.learning_steps)
            writer.add_scalar('stats/acc_exp', acc_exp, self.learning_steps)
    
    def save_reward_function(self, path):
        fname = os.path.join(path, 'reward.pt')
        os.makedirs(path, exist_ok=True)
        torch.save({
            'g': self.disc.g.state_dict(),
            'h': self.disc.h.state_dict()
            }, fname)
        