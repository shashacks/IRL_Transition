import os
from tqdm import tqdm
import numpy as np
import torch
import random
import gym
import time

from irl.buffer import Buffer, StartBuffer

from algo.ppo.ppo import PPO
from algo.trpo.trpo import TRPO
import algo.ppo.core as core_ppo
import algo.trpo.core as core_trpo

import tensorflow as tf
import h5py

import baselines.common.tf_util as U
from baselines.common import set_global_seeds
from baselines import logger
from baselines.common.atari_wrappers import TransitionEnvWrapper

from rl.mlp_policy import MlpPolicy
from rl.util import make_env

        # initialize models
def load_model(load_model_path, var_list=None):
    if os.path.isdir(load_model_path):
        ckpt_path = tf.train.latest_checkpoint(load_model_path)
    else:
        ckpt_path = load_model_path
    if ckpt_path:
        U.load_state(ckpt_path, var_list)
    return ckpt_path

def tensor_description(var):
        description = '({} [{}])'.format(
            var.dtype.name, 'x'.join([str(size) for size in var.get_shape()]))
        return description

def set_seed(seed, env, p_env):
    # Random seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    random.seed(seed)   
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)       # for multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    env.seed(seed)
    p_env.seed(seed)

def collect_demo(args):

    sess = U.single_threaded_session(gpu=False)
    sess.__enter__()

    env_name = args.env
    p_env_name = args.primitive_env
    env = make_env(env_name)
    p_env = make_env(p_env_name)
    set_seed(args.seed, env, p_env)
    
    env.unwrapped.set_curbs_x_randomness_for_irl()

    set_global_seeds(args.seed)

        # build vanilla TRPO
    policy = MlpPolicy(
        env=p_env,
        name="%s/pi" % p_env_name,
        ob_env_name=p_env_name,
        config=args)

    old_policy = MlpPolicy(
        env=p_env,
        name="%s/oldpi" % p_env_name,
        ob_env_name=p_env_name,
        config=args)
    
    networks = []
    networks.append(policy)
    networks.append(old_policy)

    var_list = []
    for network in networks:
        var_list += network.get_variables()


    if True:
        for var in var_list:
            logger.info('{} {}'.format(var.name, tensor_description(var)))

    ckpt_path = load_model(args.primitive_path, var_list)

    obs_shape = env.observation_space.shape[0]
    act_shape = env.action_space.shape[0]
    buffer_size = args.exp_buffer_size
    if args.env == 'Walker2dCrawl-v1':
        obs_shape += 2
    b_sim_info = None
    if args.front:
        buffer_size = args.exp_buffer_size // 100
        start_buffer = StartBuffer(
            buffer_size= buffer_size,
            time=1,
            qpos=env.unwrapped.model.nq,
            qvel=env.unwrapped.model.nv,
            qact=act_shape,
            obstacle_pos=1,
            device=torch.device('cpu')
        )
        o = env.reset()
        i = 0
        while i < buffer_size:
            # env.render()
            # print(env.unwrapped.get_x_pos())
            sim_info = env.unwrapped.get_sim_state()
            if args.primitive_env == 'Walker2dForward-v1' and args.env == 'Walker2dJump-v1':
                o = o[:-2]
            a, vpred = policy.act(o, False)
            n_o, r, d, info = env.step(a)
            if args.complex_task == 'hurdle':
                if args.primitive_env == 'Walker2dForward-v1':
                    flag = env.unwrapped.is_boundary_walk_front()
                elif args.primitive_env == 'Walker2dJump-v1':
                    flag = env.unwrapped.is_boundary_jump_front()
            elif args.complex_task == 'obstacle':
                if args.env == 'Walker2dJump-v1':
                    if args.primitive_env == 'Walker2dForward-v1':
                        flag = env.unwrapped.is_boundary_walk_front_for_obstacle()
                    elif args.primitive_env == 'Walker2dJump-v1':
                        flag = env.unwrapped.is_boundary_jump_front_for_obstacle()
                elif args.env == 'Walker2dCrawl-v1':
                    if args.primitive_env == 'Walker2dForward-v1':
                        flag = env.unwrapped.is_boundary_walk_front_for_obstacle()
                    elif args.primitive_env == 'Walker2dCrawl-v1':
                        flag = env.unwrapped.is_boundary_crawl_front_for_obstacle()
            elif args.complex_task == 'patrol':
                if args.env == 'Walker2dForward-v1':
                    flag = env.unwrapped.is_boundary_forward_front_for_patrol()
                elif args.env == 'Walker2dBalance-v1':
                    flag = env.unwrapped.is_boundary_balance_front_for_patrol()
                elif args.env == 'Walker2dBackward-v1':
                    flag = env.unwrapped.is_boundary_backward_front_for_patrol()
                        


            # print(flag, env.unwrapped.get_x_pos(), env.unwrapped.get_curb_pos())
            if flag == 0 and b_sim_info != None:
                if i % 100 == 0:
                    print(f'(front: {start_buffer._p} / {buffer_size})')
                i = i + 1
                start_buffer.append(np.array([b_sim_info['state'][0]]), b_sim_info['state'][1], b_sim_info['state'][2], b_ac, np.array([b_sim_info['obstacle_pos']]))
                flag = -1

            o = n_o
            b_sim_info = sim_info
            b_ac = a

            if d or flag == -1:
                o = env.reset()
                b_sim_info = None


        start_buffer.save(os.path.join(
            'data',
            'exp_demo',
            args.primitive_env + '_' +  args.suffix,
            'exp_demo_start.pth'
            ))

    else: 
        buffer = Buffer(
            buffer_size= buffer_size,
            state_shape=obs_shape,
            action_shape=act_shape,
            device=torch.device('cpu')
        )

        total_return = 0.0
        num_episodes = 0

        o = env.reset()
        episode_return = 0.0
        i = 0
        states = []
        while i < buffer_size:
            env.render()
            print(env.unwrapped.get_x_pos())
            sim_info = env.unwrapped.get_sim_state()
            if args.primitive_env == 'Walker2dForward-v1' and args.env == 'Walker2dJump-v1':
                o = o[:-2]
            o = np.array(o, dtype=np.float32)
            if args.primitive_env == 'Walker2dCrawl-v1' and args.env == 'Walker2dCrawl-v1':
                dist = env.unwrapped.get_dist()

            a, vpred = policy.act(o, False)
            n_o, r, d, info = env.step(a)
            n_o = np.array(n_o, dtype=np.float32)
            if args.primitive_env == 'Walker2dCrawl-v1' and args.env == 'Walker2dCrawl-v1':
                n_dist = env.unwrapped.get_dist()

            if args.primitive_env == 'Walker2dForward-v1' and args.env == 'Walker2dJump-v1':
                o = np.append(o, np.array([5.1, 5.2], dtype=np.float32))
                n_o = np.append(n_o[:-2], np.array([5.1, 5.2], dtype=np.float32))
            elif args.primitive_env == 'Walker2dForward-v1' and args.env == 'Walker2dCrawl-v1':
                o = np.append(o, np.array([5.1, 6.0], dtype=np.float32))
                n_o = np.append(n_o, np.array([5.1, 6.0], dtype=np.float32))
            elif args.primitive_env == 'Walker2dCrawl-v1' and args.env == 'Walker2dCrawl-v1':
                o = np.append(o, np.array(dist, dtype=np.float32))
                n_o = np.append(n_o, np.array(n_dist, dtype=np.float32))

            if args.complex_task == 'hurdle':
                if args.primitive_env == 'Walker2dForward-v1':
                    flag = env.unwrapped.is_boundary_walk_rear()
                elif args.primitive_env == 'Walker2dJump-v1':
                    flag = env.unwrapped.is_boundary_jump_rear()
            elif args.complex_task == 'obstacle':
                if args.env == 'Walker2dJump-v1':
                    if args.primitive_env == 'Walker2dForward-v1':
                        flag = env.unwrapped.is_boundary_walk_rear()
                    elif args.primitive_env == 'Walker2dJump-v1':
                        flag = env.unwrapped.is_boundary_jump_rear_for_obstacle()
                elif args.env == 'Walker2dCrawl-v1':
                    if args.primitive_env == 'Walker2dForward-v1':
                        flag = env.unwrapped.is_boundary_walk_rear_for_obstacle()
                    elif args.primitive_env == 'Walker2dCrawl-v1':
                        flag = env.unwrapped.is_boundary_crawl_rear_for_obstacle()
            elif args.complex_task == 'patrol':
                if args.env == 'Walker2dForward-v1':
                    flag = env.unwrapped.is_boundary_forward_rear_for_patrol()
                elif args.env == 'Walker2dBalance-v1':
                    flag = env.unwrapped.is_boundary_balance_rear_for_patrol()
                elif args.env == 'Walker2dBackward-v1':
                    flag = env.unwrapped.is_boundary_backward_rear_for_patrol()

  
            # print(flag, env.unwrapped.get_x_pos(), env.unwrapped.get_curb_pos(), env.unwrapped.get_curb_pos() - env.unwrapped.get_x_pos())
            # print(dist, n_dist)
            if flag == 0:
                states.append(o)
                buffer.append(o, a, r, d, n_o)
                # print('append')
                if i % 1000 == 0:
                    print(f'(rear: {i}, {buffer_size})')
                    print(env.unwrapped.get_curbs_x_randomness())
                i = i + 1
                episode_return += r
            if d or flag == -1:
                num_episodes += 1
                total_return += episode_return
                o = env.reset()
                episode_return = 0.0
  
            if args.env == 'Walker2dCrawl-v1':
                o = n_o[:-2]
            else:
                o = n_o

        states = np.array(states, dtype=np.float32)
        print(np.mean(states, axis=0))
        print(np.std(states, axis=0))
        print(f'Mean return of the expert is {total_return / num_episodes}')
        
        buffer.save(os.path.join(
            'data',
            'exp_demo',
            args.primitive_env + '_' +  args.suffix,
            'exp_demo.pth'
        ))

