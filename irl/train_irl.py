import os
import argparse
from datetime import datetime
import torch

import gym

from irl.buffer import SerializedStartBuffer, SerializedBuffer
from irl.algo import ALGOS
from irl.trainer import Trainer


def train_irl(args):
    env = gym.make(args.env)
    env_test = gym.make(args.env)
    if args.complex_task == 'obstacle' or args.complex_task == 'hurdle': 
        env.unwrapped.set_curbs_x_randomness_for_irl()
        env_test.unwrapped.set_curbs_x_randomness_for_irl()
    elif args.complex_task == 'pick':
        env.unwrapped.set_randomness_for_irl()
        env_test.unwrapped.set_randomness_for_irl()

    if args.env == 'Walker2dCrawl-v1':
        env_obs_shape = tuple(map(sum, zip((2,), env.observation_space.shape)))
    else:
        env_obs_shape = env.observation_space.shape
    
    start_buffer_exp = SerializedStartBuffer(
        path=os.path.join(args.exp_data_path_1, 'exp_demo_start.pth'),
        device=torch.device("cpu")
    )

    buffer_exp = SerializedBuffer(
        path=os.path.join(args.exp_data_path_2, 'exp_demo.pth'),
        device=torch.device("cpu")
    )

    algo = ALGOS[args.irl_algo](
        buffer_exp=buffer_exp,
        args = args,
        start_buffer_exp= start_buffer_exp,
        state_shape=env_obs_shape,
        action_shape=env.action_space.shape,
        device=torch.device("cpu"),
        seed=args.seed,
        rollout_length=50000,
        front=args.front
    )

    trainer = Trainer(
        env=env,
        env_test=env_test,
        algo=algo,
        seed=args.seed,
        args=args,
        start_buffer_exp= start_buffer_exp
    )


    trainer.train()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    args = p.parse_args()
    run(args)