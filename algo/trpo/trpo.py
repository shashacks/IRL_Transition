import sys
import os
import os.path as osp
from mpi4py import MPI
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import gym
import time
import random
import algo.trpo.core as core
import spinup.utils.dataset as dataset
from algo.trpo.buffer import GAEBuffer
from spinup.utils.logx import EpochLogger
from spinup.utils.mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads
from spinup.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs


class TRPO:
    def __init__(self, args, logger_kwargs):
        self.env = gym.make(args.env)
        self.args = args
        self.logger = EpochLogger(**logger_kwargs)
        self.seed = args.seed + 10000 * proc_id()
        self.set_seed()
        self.num_rollouts = args.num_rollouts
        self.epochs = args.epochs
        self.gamma = args.gamma
        self.delta = args.delta
        self.clip_ratio = args.clip_ratio
        self.vf_lr = args.ppo_vf_lr
        self.train_v_iters = args.trpo_train_v_iters
        self.batch_size = args.trpo_batchsize
        self.damping_coeff = args.damping_coeff
        self.cg_iters = args.cg_iters
        self.eps = 1e-8
        self.backtrack_iters = args.backtrack_iters
        self.backtrack_coeff = args.backtrack_coeff
        self.lam = args.lam
        self.save_freq = args.save_freq
        self.obs_dim = self.env.observation_space.shape
        self.act_dim = self.env.action_space.shape
        self.max_steps = args.max_steps
        self.ac = core.MLPActorCritic(self.env.observation_space, self.env.action_space, 
            hidden_sizes = [args.primitive_hid_size] * args.primitive_hid_layer)
        
        sync_params(self.ac)

        # Count variables
        var_counts = tuple(core.count_vars(module) for module in [self.ac.pi, self.ac.v])
        self.logger.log('\nNumber of parameters: \t pi: %d, \t v: %d\n'%var_counts)



        if proc_id() == 0:
            self.writer = SummaryWriter(log_dir=self.logger.output_dir + '/tensorboard')

        # Set up experience buffer
        self.buf = GAEBuffer(self.obs_dim, self.act_dim, self.num_rollouts, self.gamma, self.lam)

        self.vf_optimizer = torch.optim.Adam(self.ac.v.parameters(), lr=self.vf_lr)
        self.o, self.ep_ret, self.ep_len = self.env.reset(), 0, 0
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

    # Set up function for computing PPO policy loss
    def compute_loss_pi(self, data):
        obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']
        # Policy loss
        _, logp = self.ac.pi(obs, act)
        ratio = torch.exp(logp - logp_old)
        loss_pi = -(ratio * adv).mean()
        return loss_pi

    # Set up function for computing value loss
    def compute_loss_v(self, data):
        obs, ret = data['obs'], data['ret']
        return ((self.ac.v(obs) - ret) ** 2).mean()

    def compute_loss_v_batch(self, obs, ret):
        obs = torch.as_tensor(obs, dtype=torch.float32)
        ret = torch.as_tensor(ret, dtype=torch.float32)
        return ((self.ac.v(obs) - ret) ** 2).mean()

    def compute_kl(self, data, old_pi):
        obs, act = data['obs'], data['act']
        pi, _ = self.ac.pi(obs, act)
        kl_loss = torch.distributions.kl_divergence(pi, old_pi).mean()
        return kl_loss
    
    @torch.no_grad()
    def compute_kl_loss_pi(self, data, old_pi):
        obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']
        # Policy loss
        pi, logp = self.ac.pi(obs, act)
        ratio = torch.exp(logp - logp_old)
        loss_pi = -(ratio * adv).mean()
        kl_loss = torch.distributions.kl_divergence(pi, old_pi).mean()
        return loss_pi, kl_loss

    def hessian_vector_product(self, data, old_pi, v):
        kl = self.compute_kl(data, old_pi)

        grads = torch.autograd.grad(kl, self.ac.pi.parameters(), create_graph=True)
        flat_grad_kl = core.flat_grads(grads)

        kl_v = (flat_grad_kl * v).sum()
        grads = torch.autograd.grad(kl_v, self.ac.pi.parameters())
        flat_grad_grad_kl = core.flat_grads(grads)

        return flat_grad_grad_kl + v * self.damping_coeff

    def get_rollouts(self, epoch=0):
        success_count_list = []
        ep_ret_list = []
        ep_len_list = []
        for t in range(self.num_rollouts):
            a, v, logp = self.ac.step(torch.as_tensor(self.o, dtype=torch.float32))

            next_o, r, d, info = self.env.step(a)
            self.ep_ret += r
            self.ep_len += 1

            # save and log
            self.buf.store(self.o, a, r, v, logp)
            self.logger.store(VVals=v)
            
            # Update obs (critical!)
            self.o = next_o

            timeout = self.ep_len == self.max_steps
            terminal = d or timeout
            epoch_ended = t==self.num_rollouts-1

            if terminal or epoch_ended:

                if epoch_ended and not(terminal):
                    print('Warning: trajectory cut off by epoch at %d steps.'%self.ep_len, flush=True)
                # if trajectory didn't reach terminal state, bootstrap value target
                if timeout or epoch_ended:
                    _, v, _ = self.ac.step(torch.as_tensor(self.o, dtype=torch.float32))
                else:
                    v = 0
                self.buf.finish_path(v)
                if terminal:
                    # only save EpRet / EpLen if trajectory finished
                    success_count_list.append(info['success_count'])
                    ep_ret_list.append(self.ep_ret)
                    ep_len_list.append(self.ep_len)
                    self.logger.store(EpRet=self.ep_ret, EpLen=self.ep_len, EpSuccessCount=info['success_count'])

                if epoch_ended:
                    self.tensorboard_scalars['EpRet'] = sum(ep_ret_list) / len(ep_ret_list) 
                    self.tensorboard_scalars['EpLen'] = sum(ep_len_list) / len(ep_len_list) 
                    self.tensorboard_scalars['EpSuccessCount'] = sum(success_count_list) / len(success_count_list) 
                    self.tensorboard_scalars['MaxEpRet'] = max(ep_ret_list)

                self.o, self.ep_ret, self.ep_len = self.env.reset(), 0, 0


    def update(self, data):
        # compute old pi distribution
        obs, act = data['obs'], data['act']
        with torch.no_grad():
            old_pi, _ = self.ac.pi(obs, act)

        pi_loss = self.compute_loss_pi(data)
        pi_l_old = pi_loss.item()
        v_l_old = self.compute_loss_v(data).item()

        grads = core.flat_grads(torch.autograd.grad(pi_loss, self.ac.pi.parameters()))

        # Core calculations for TRPO or NPG
        Hx = lambda v: self.hessian_vector_product(data, old_pi, v)
        x = core.conjugate_gradients(Hx, grads, self.cg_iters)

        alpha = torch.sqrt(2.0 * self.delta / torch.abs((torch.matmul(x, Hx(x)) + self.eps)))

        old_params = core.get_flat_params_from(self.ac.pi)

        def set_and_eval(step):
            new_params = old_params - alpha * x * step
            core.set_flat_params_to(self.ac.pi, new_params)
            loss_pi, kl_loss = self.compute_kl_loss_pi(data, old_pi)
            return kl_loss.item(), loss_pi.item()

        if self.args.primitive_algo == 'npg':
            # npg has no backtracking or hard kl constraint enforcement
            kl, pi_l_new = set_and_eval(step=1.)

        elif self.args.primitive_algo == 'trpo':
            # trpo augments npg with backtracking line search, hard kl
            for j in range(self.backtrack_iters):
                kl, pi_l_new = set_and_eval(step=self.backtrack_coeff *pow(0.5 , j))
                if kl <= self.delta * 1.5 and pi_l_new <= pi_l_old:
                    self.logger.log('Accepting new params at step %d of line search.' % j)
                    self.logger.store(BacktrackIters=j)
                    break

                if j == self.backtrack_iters - 1:
                    self.logger.log('Line search failed! Keeping old params.')
                    self.logger.store(BacktrackIters=j)
                    kl, pi_l_new = set_and_eval(step=0.)

        # # Value function learning
        # for i in range(self.train_v_iters):
        #     self.vf_optimizer.zero_grad()
        #     loss_v = self.compute_loss_v(data)
        #     loss_v.backward()
        #     mpi_avg_grads(self.ac.v)  # average grads across MPI processes
        #     self.vf_optimizer.step()
        d = dataset.Dataset(dict(obs=data['obs'] , ret=data['ret']), shuffle=True)

        for _ in range(self.train_v_iters):
            for batch in d.iterate_once(self.batch_size):
                self.vf_optimizer.zero_grad()
                loss_v = self.compute_loss_v_batch(batch['obs'], batch['ret'])
                loss_v.backward()
                mpi_avg_grads(self.ac.v)  # average grads across MPI processes
                self.vf_optimizer.step()

        # Log changes from update
        self.tensorboard_scalars['Entropy'] = self.ac.pi.entropy().mean().item()
        self.tensorboard_scalars['KL'] = kl
        self.tensorboard_scalars['LossPi'] = pi_l_new
        self.tensorboard_scalars['LossV'] = loss_v.item()
        self.tensorboard_scalars['DeltaLossPi'] = (pi_l_new - pi_l_old)
        self.tensorboard_scalars['DeltaLossV'] = (loss_v.item() - v_l_old)

        # Log changes from update
        self.logger.store(LossPi=pi_l_old, LossV=v_l_old, KL=kl,
                     DeltaLossPi=(pi_l_new - pi_l_old),
                     DeltaLossV=(loss_v.item() - v_l_old))

    def train(self):
        # Main loop: collect experience in env and update/log each epoch
        for epoch in range(self.epochs):
            self.get_rollouts(epoch)

            data = self.buf.get()
            self.ac.update(data['obs'])
            # Perform PPO update!
            self.update(data)
            

            # Save model
            if ((epoch % self.save_freq == 0) or (epoch == self.epochs-1)) and proc_id() == 0:
                self.save_model()

            if proc_id() == 0:
                self.write_tensorboard(epoch)

            # Log info about epoch
            self.logger.log_tabular('Epoch', epoch)
            self.logger.log_tabular('EpRet', with_min_and_max=True)
            self.logger.log_tabular('EpLen', with_min_and_max=True)
            self.logger.log_tabular('EpSuccessCount', with_min_and_max=True)
            self.logger.log_tabular('VVals', with_min_and_max=True)
            self.logger.log_tabular('TotalEnvInteracts', (epoch + 1) * self.num_rollouts)
            self.logger.log_tabular('LossPi', average_only=True)
            self.logger.log_tabular('LossV', average_only=True)
            self.logger.log_tabular('DeltaLossPi', average_only=True)
            self.logger.log_tabular('DeltaLossV', average_only=True)
            self.logger.log_tabular('KL', average_only=True)
            if self.args.primitive_algo == 'trpo':
                self.logger.log_tabular('BacktrackIters', average_only=True)
            self.logger.log_tabular('Time', time.time() - self.start_time)
            self.logger.dump_tabular()
    
    def save_model(self, sub=''):
        fpath = self.logger.output_dir
        fpath = osp.join(osp.join(fpath, 'pyt_save'), sub)
        fname = osp.join(fpath, 'model.pt')
    
        os.makedirs(fpath, exist_ok=True)
        torch.save({
            'pi_rms_mean': self.ac.pi.rms_mean,
            'pi_rms_std': self.ac.pi.rms_std,
            'pi_rms_count': self.ac.pi.rms_count,
            'v_rms_mean': self.ac.v.rms_mean,
            'v_rms_std': self.ac.v.rms_std,
            'v_rms_count': self.ac.v.rms_count,
            'model': self.ac.state_dict(),
            }, fname)

    def write_tensorboard(self, epoch):
        self.writer.add_scalar(self.args.env + '/AverageEpRet', self.tensorboard_scalars['EpRet'], epoch)
        self.writer.add_scalar(self.args.env + '/MaxEpRet', self.tensorboard_scalars['MaxEpRet'], epoch)
        self.writer.add_scalar(self.args.env + '/EpLen', self.tensorboard_scalars['EpLen'], epoch)
        self.writer.add_scalar(self.args.env + '/Entropy', self.tensorboard_scalars['Entropy'], epoch)
        self.writer.add_scalar(self.args.env + '/KL', self.tensorboard_scalars['KL'], epoch)
        self.writer.add_scalar(self.args.env + '/LossPi', self.tensorboard_scalars['LossPi'], epoch) 
        self.writer.add_scalar(self.args.env + '/LossV', self.tensorboard_scalars['LossV'], epoch)
        self.writer.add_scalar(self.args.env + '/DeltaLossPi', self.tensorboard_scalars['DeltaLossPi'], epoch) 
        self.writer.add_scalar(self.args.env + '/DeltaLossV', self.tensorboard_scalars['DeltaLossV'], epoch)
        self.writer.add_scalar(self.args.env +'/EpSuccessCount', self.tensorboard_scalars['EpSuccessCount'], epoch)

