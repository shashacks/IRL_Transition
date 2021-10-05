import os

from time import time, sleep
from datetime import timedelta
from torch.utils.tensorboard import SummaryWriter

import torch
import random
import numpy as np

class Trainer:
    # algo airl
    def __init__(self, env, env_test, algo, args, start_buffer_exp, seed=0, num_steps=10000000,
                 eval_interval=10000, num_eval_episodes=5, train_with_learned_reward=False):
        super().__init__()

        self.seed = seed

        # Env to collect samples.
        self.env = env

        # Env for evaluation.
        self.env_test = env_test

        self.set_seed()

        self.args = args
        self.algo = algo
        self.log_dir = os.path.join('data/transition', args.env_1 + '_' + args.env_2)

        # Log setting.
        self.summary_dir = os.path.join(self.log_dir, 'summary')
        self.writer = SummaryWriter(log_dir=self.summary_dir)
        self.model_dir = self.log_dir
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        # Other parameters.
        self.num_steps = num_steps
        self.eval_interval = eval_interval
        self.num_eval_episodes = num_eval_episodes
        
        self.train_with_learned_reward = train_with_learned_reward
        self.start_buffer_exp = start_buffer_exp

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
        self.env_test.seed(2**31-self.seed)

    def train(self):
        # Time to start training.
        self.start_time = time()
        # Episode's timestep.
        t = 0
        # Initialize the environment.
        state = self.env.reset()
        start_sample = self.start_buffer_exp.sample()   

            
        state = self.env.unwrapped.rollback(start_sample[0].cpu().detach().numpy(), 
            start_sample[1].cpu().detach().numpy(),
            start_sample[2].cpu().detach().numpy())

        for step in range(1, self.num_steps + 1):
            # Pass to the algorithm to update state and episode timestep.
            state, t = self.algo.step(self.env, state, t, step)
            # self.env.render()
            # sleep(1e-2)

            # Update the algorithm whenever ready.
            if self.algo.is_update(step):
                self.algo.update(self.writer)

            # Evaluate regularly.
            if step % self.eval_interval == 0:
                self.evaluate(step)

            if step % (self.eval_interval * 100) == 0:
                self.algo.save_reward_function(self.model_dir)
                self.algo.save_models(
                    os.path.join(self.model_dir, f'step{step}'))
        # Wait for the logging to be finished.
        sleep(10)

    def evaluate(self, step):
        mean_return = 0.0

        for _ in range(self.num_eval_episodes):
            state = self.env_test.reset()
            start_sample = self.start_buffer_exp.sample()            
            state = self.env_test.unwrapped.rollback(start_sample[0].cpu().detach().numpy(), 
                start_sample[1].cpu().detach().numpy(),
                start_sample[2].cpu().detach().numpy())
            episode_return = 0.0
            done = False

            if self.args.front:
                flag = self.env_test.unwrapped.is_transition_boundary()
            else:
                flag = self.env_test.unwrapped.is_transition_boundary_rear()
            while (not done) and (not flag ==-1):
                if self.args.front:
                    flag = self.env_test.unwrapped.is_transition_boundary()
                else:
                    flag = self.env_test.unwrapped.is_transition_boundary_rear()
                action = self.algo.ac.act(torch.as_tensor(state, dtype=torch.float32))
                state, reward, done, info = self.env_test.step(action)
                episode_return += reward

                if step >= 100000:
                    self.env_test.render()
                    print('xpos:', self.env_test.unwrapped.get_x_pos(), 'alive_reward:', info['alive_reward'], 'ctrl_reward:', info['ctrl_reward'], 'collision_penalty:', info['collision_penalty'], 'x_vel_reward:', info['x_vel_reward']
                    , 'angle_reward:', info['angle_reward'], 'height_reward:', info['height_reward'], 'foot_reward:', info['foot_reward'], 'jump_reward:', info['jump_reward'], 'pass_reward:', info['pass_reward'])


            mean_return += episode_return / self.num_eval_episodes

        self.writer.add_scalar('return/test', mean_return, step)
        print(f'Num steps: {step:<6}   '
              f'Return: {mean_return:<5.1f}   '
              f'Time: {self.time}   '
              f'env: {self.env.unwrapped.get_curbs_x_randomness()}   '
              f'env_test: {self.env_test.unwrapped.get_curbs_x_randomness()}')

    @property
    def time(self):
        return str(timedelta(seconds=int(time() - self.start_time)))
