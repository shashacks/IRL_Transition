import sys
import os
import os.path as osp
from mpi4py import MPI
import numpy as np
import torch
import torch.nn as nn
import random
import gym

from config import argparser
from spinup.utils.logx import EpochLogger
from spinup.utils.mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads
from spinup.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs
from algo.ppo.ppo import PPO
from algo.trpo.trpo import TRPO
from algo.sac.sac import SAC
from irl.collect_demo import collect_demo
from irl.collect_demo_arm import collect_demo_arm
from irl.train_irl import train_irl

from q_network.train_q_network_hurdle import train_q_network_hurdle
from q_network.train_q_network_obstacle import train_q_network_obstacle
from q_network.train_q_network_pick import train_q_network_pick
from q_network.train_q_network_serve import train_q_network_serve
from q_network.train_q_network_patrol import train_q_network_patrol
from q_network.train_q_network_catch import train_q_network_catch

from q_network.test_q_network_hurdle import test_q_network_hurdle
from q_network.test_q_network_obstacle import test_q_network_obstacle
from q_network.test_q_network_pick import test_q_network_pick
from q_network.test_q_network_serve import test_q_network_serve
from q_network.test_q_network_patrol import test_q_network_patrol
from q_network.test_q_network_catch import test_q_network_catch

def train_primitive(args, logger_kwargs):

    if args.primitive_algo == 'ppo':
        algo = PPO(args, logger_kwargs)
    elif args.primitive_algo == 'trpo':
        algo = TRPO(args, logger_kwargs)
            
    algo.train()
    

def main():
    args = argparser()
    mpi_fork(args.mpi)  # run parallel code with mpi
    from spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    # Special function to avoid certain slowdowns from PyTorch + MPI combo.
    setup_pytorch_for_mpi()
    if args.train:
        if args.hrl:
            print(f"Start training a Q-network")
            if args.complex_task == 'obstacle':
                train_q_network_obstacle(args)
            elif args.complex_task == 'hurdle':
                train_q_network_hurdle(args)
            elif args.complex_task == 'pick':
                train_q_network_pick(args)
            elif args.complex_task == 'serve':
                train_q_network_serve(args)
            elif args.complex_task == 'patrol':
                train_q_network_patrol(args)
            elif args.complex_task == 'catch':
                train_q_network_catch(args)
            
            print(f"Done training a Q-network")
            
        elif args.irl_training:
            print("Start IRL training")  
            train_irl(args)
            print("Done IRL training")  
        else:
            print("Start training primitive policy")  
            train_primitive(args, logger_kwargs)
            print("Done training primitive policy")                  
    else:
        if args.hrl:
            print(f"Start test hrl result")
            if args.complex_task == 'hurdle':
                test_q_network_hurdle(args)
            elif args.complex_task == 'obstacle':
                test_q_network_obstacle(args)
            elif args.complex_task == 'pick':
                test_q_network_pick(args)
            elif args.complex_task == 'serve':
                test_q_network_serve(args)
            elif args.complex_task == 'patrol':
                test_q_network_patrol(args)
            elif args.complex_task == 'catch':
                test_q_network_catch(args) 
            
            print(f"Done test hrl result")
        if args.collect_exp_data:
            print(f"Start collecting {args.env} expert demonstrations")
            if args.complex_task == 'hurdle' or args.complex_task == 'obstacle' or args.complex_task == 'patrol':
                collect_demo(args)
            elif args.complex_task == 'pick' or args.complex_task == 'serve' or args.complex_task == 'catch':
                collect_demo_arm(args)
            print(f"Done collecting {args.env} expert demonstrations")
        # else: rendering?

if __name__ == '__main__':
    main()
