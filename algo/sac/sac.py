import sys
import os
import os.path as osp
from mpi4py import MPI
from copy import deepcopy
import itertools
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import gym
import time
import random
import algo.sac.core as core
import spinup.utils.dataset as dataset
from algo.sac.buffer import ReplayBuffer
from spinup.utils.logx import EpochLogger
from spinup.utils.mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads
from spinup.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs


class SAC:
    def __init__(self, args, logger_kwargs):
        self.env = gym.make(args.env)
        self.args = args
        self.logger = EpochLogger(**logger_kwargs)
        self.seed = args.seed + 10000 * proc_id()
        self.set_seed()
        self.steps_per_epoch = args.sac_steps_per_epoch
        self.epochs = args.epochs
        self.replay_size = args.sac_replay_size
        self.gamma = args.sac_gamma
        self.polyak = args.sac_polyak
        self.lr = args.sac_lr
        self.alpha = args.sac_alpha
        self.batch_size = args.sac_batchsize
        self.start_steps = args.sac_start_steps
        self.update_after = args.sac_update_after
        self.update_every = args.sac_update_every
        self.max_ep_len = args.sac_max_ep_len
        self.save_freq = args.sac_save_freq
        self.obs_dim = self.env.observation_space.shape
        self.act_dim = self.env.action_space.shape[0]
        self.act_limit = self.env.action_space.high[0]

        # Create actor-critic module and target networks
        self.ac = core.MLPActorCritic(self.env.observation_space, self.env.action_space, 
                    hidden_sizes = [256] * 2)
        self.ac_targ = deepcopy(self.ac)
        sync_params(self.ac)
        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.ac_targ.parameters():
            p.requires_grad = False
            
        # List of parameters for both Q-networks (save this for convenience)
        self.q_params = itertools.chain(self.ac.q1.parameters(), self.ac.q2.parameters())

        # Experience buffer
        self.replay_buffer = ReplayBuffer(obs_dim=self.obs_dim, act_dim=self.act_dim, size=self.replay_size)

        self.pi_optimizer = torch.optim.Adam(self.ac.pi.parameters(), lr=self.lr)
        self.q_optimizer = torch.optim.Adam(self.q_params, lr=self.lr)


        # Count variables (protip: try to get a feel for how different size networks behave!)
        self.var_counts = tuple(core.count_vars(module) for module in [self.ac.pi, self.ac.q1, self.ac.q2])
        self.logger.log('\nNumber of parameters: \t pi: %d, \t q1: %d, \t q2: %d\n'%self.var_counts)

        self.writer = SummaryWriter(log_dir=self.logger.output_dir + '/tensorboard')


        # Prepare for interaction with environment
        self.total_steps = self.steps_per_epoch * self.epochs
        self.start_time = time.time()
        
        self.tensorboard_scalars = {}

    def set_seed(self):
        # Random seed
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        
        random.seed(self.seed)   
        torch.cuda.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)       # for multi-GPU.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        self.env.seed(self.seed)

    # Set up function for computing SAC Q-losses
    def compute_loss_q(self, data):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']

        q1 = self.ac.q1(o,a)
        q2 = self.ac.q2(o,a)

        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            a2, logp_a2 = self.ac.pi(o2)

            # Target Q-values
            q1_pi_targ = self.ac_targ.q1(o2, a2)
            q2_pi_targ = self.ac_targ.q2(o2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + self.gamma * (1 - d) * (q_pi_targ - self.alpha * logp_a2)

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup)**2).mean()
        loss_q2 = ((q2 - backup)**2).mean()
        loss_q = loss_q1 + loss_q2

        # Useful info for logging
        q_info = dict(Q1Vals=q1.detach().numpy(),
                      Q2Vals=q2.detach().numpy())

        return loss_q, q_info

    # Set up function for computing SAC pi loss
    def compute_loss_pi(self, data):
        o = data['obs']
        pi, logp_pi = self.ac.pi(o)
        q1_pi = self.ac.q1(o, pi)
        q2_pi = self.ac.q2(o, pi)
        q_pi = torch.min(q1_pi, q2_pi)

        # Entropy-regularized policy loss
        loss_pi = (self.alpha * logp_pi - q_pi).mean()

        # Useful info for logging
        pi_info = dict(LogPi=logp_pi.detach().numpy())

        return loss_pi, pi_info

    def update(self, data):
        # First run one gradient descent step for Q1 and Q2
        self.q_optimizer.zero_grad()
        loss_q, q_info = self.compute_loss_q(data)
        loss_q.backward()
        self.q_optimizer.step()

        # Record things
        self.logger.store(LossQ=loss_q.item(), **q_info)
        self.tensorboard_scalars['LossQ'] = loss_q.item()

        # Freeze Q-networks so you don't waste computational effort 
        # computing gradients for them during the policy learning step.
        for p in self.q_params:
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        self.pi_optimizer.zero_grad()
        loss_pi, pi_info = self.compute_loss_pi(data)
        loss_pi.backward()
        self.pi_optimizer.step()

        # Unfreeze Q-networks so you can optimize it at next DDPG step.
        for p in self.q_params:
            p.requires_grad = True

        # Record things
        self.logger.store(LossPi=loss_pi.item(), **pi_info)
        self.tensorboard_scalars['LossPi'] = loss_pi.item()

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(self.ac.parameters(), self.ac_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)

    def get_action(self, o, deterministic=False):
        return self.ac.act(torch.as_tensor(o, dtype=torch.float32), 
                      deterministic)
    
    def train(self):
        success_count_list = []
        ep_ret_list = []
        ep_len_list = []
        x_pos_list = []

        o, ep_ret, ep_len = self.env.reset(), 0, 0
        for t in range(self.total_steps):
            
            # Until start_steps have elapsed, randomly sample actions
            # from a uniform distribution for better exploration. Afterwards, 
            # use the learned policy. 
            if t > self.start_steps:
                a = self.get_action(o)
            else:
                a = self.env.action_space.sample()

            # Step the envf
            self.ac.update(o)
            o2, r, d, info = self.env.step(a)
            ep_ret += r
            ep_len += 1

            # Ignore the "done" signal if it comes from hitting the time
            # horizon (that is, when it's an artificial terminal signal
            # that isn't based on the agent's state)
            d = False if ep_len==self.max_ep_len else d

            # Store experience to replay buffer
            self.replay_buffer.store(o, a, r, o2, d)

            # Super critical, easy to overlook step: make sure to update 
            # most recent observation!
            o = o2

            # End of trajectory handling
            if d or (ep_len == self.max_ep_len):
                success_count_list.append(info['success_count'])
                ep_ret_list.append(ep_ret)
                ep_len_list.append(ep_len)
                x_pos_list.append(info['x_pos'])
                self.logger.store(EpRet=ep_ret, EpLen=ep_len, EpSuccessCount=info['success_count'], XPos=info['x_pos'])

                o, ep_ret, ep_len = self.env.reset(), 0, 0

            # Update handling
            if t >= self.update_after and t % self.update_every == 0:
                for j in range(self.update_every):
                    batch = self.replay_buffer.sample_batch(self.batch_size)
                    self.update(data=batch)

            # End of epoch handling
            if (t+1) % self.steps_per_epoch == 0:
                epoch = (t+1) // self.steps_per_epoch

                self.tensorboard_scalars['EpRet'] = sum(ep_ret_list) / len(ep_ret_list) 
                self.tensorboard_scalars['EpLen'] = sum(ep_len_list) / len(ep_len_list) 
                self.tensorboard_scalars['EpSuccessCount'] = sum(success_count_list) / len(success_count_list) 
                self.tensorboard_scalars['MaxEpRet'] = max(ep_ret_list)
                self.tensorboard_scalars['XPos'] = sum(x_pos_list) / len(x_pos_list) 
                self.tensorboard_scalars['XPosMax'] = max(x_pos_list)
                ep_ret_list.clear()
                ep_len_list.clear()
                success_count_list.clear()
                x_pos_list.clear()



                # Save model
                if (epoch % self.save_freq == 0) or (epoch == self.epochs):
                    self.save_model()

                self.write_tensorboard(epoch-1)

                # Test the performance of the deterministic version of the agent.
                # test_agent()

                # Log info about epoch
                self.logger.log_tabular('Epoch', epoch)
                self.logger.log_tabular('EpRet', with_min_and_max=True)
                self.logger.log_tabular('EpLen', average_only=True)
                self.logger.log_tabular('XPos', with_min_and_max=True)
                self.logger.log_tabular('EpSuccessCount', with_min_and_max=True)
                self.logger.log_tabular('TotalEnvInteracts', t)
                self.logger.log_tabular('Q1Vals', with_min_and_max=True)
                self.logger.log_tabular('Q2Vals', with_min_and_max=True)
                self.logger.log_tabular('LogPi', with_min_and_max=True)
                self.logger.log_tabular('LossPi', average_only=True)
                self.logger.log_tabular('LossQ', average_only=True)
                self.logger.log_tabular('Time', time.time()-self.start_time)
                self.logger.dump_tabular()

    def write_tensorboard(self, epoch):
        self.writer.add_scalar(self.args.env + '/AverageEpRet', self.tensorboard_scalars['EpRet'], epoch)
        self.writer.add_scalar(self.args.env + '/MaxEpRet', self.tensorboard_scalars['MaxEpRet'], epoch)
        self.writer.add_scalar(self.args.env + '/EpLen', self.tensorboard_scalars['EpLen'], epoch)
        self.writer.add_scalar(self.args.env + '/LossQ', self.tensorboard_scalars['LossQ'], epoch) 
        self.writer.add_scalar(self.args.env + '/LossPi', self.tensorboard_scalars['LossPi'], epoch)
        self.writer.add_scalar(self.args.env +'/EpSuccessCount', self.tensorboard_scalars['EpSuccessCount'], epoch)
        self.writer.add_scalar(self.args.env +'/XPos', self.tensorboard_scalars['XPos'], epoch)
        self.writer.add_scalar(self.args.env +'/XPosMax', self.tensorboard_scalars['XPosMax'], epoch)


    def save_model(self, sub=''):
        fpath = self.logger.output_dir
        fpath = osp.join(osp.join(fpath, 'pyt_save'), sub)
        fname = osp.join(fpath, 'model.pt')
    
        os.makedirs(fpath, exist_ok=True)
        torch.save({
            'pi_rms_mean': self.ac.pi.rms_mean,
            'pi_rms_std': self.ac.pi.rms_std,
            'pi_rms_count': self.ac.pi.rms_count,
            'model': self.ac.state_dict(),
            }, fname)