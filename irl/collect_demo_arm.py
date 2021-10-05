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

def collect_demo_arm(args):
    sess = U.single_threaded_session(gpu=False)
    sess.__enter__()

    env_name = args.env
    p_env_name = args.primitive_env
    env = make_env(env_name)
    p_env = make_env(p_env_name)
    set_seed(args.seed, env, p_env)

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
            # time.sleep(1e-2)
            sim_info = env.unwrapped.get_sim_state()
            a, vpred = policy.act(o, False)
            n_o, r, d, info = env.step(a)
            if args.complex_task == 'pick':
                flag = env.unwrapped.is_boundary_pick_front()
            elif args.complex_task == 'serve':
                flag = env.unwrapped.is_boundary_toss_front()
            elif args.complex_task == 'catch':
                flag = env.unwrapped.is_boundary_catch_front()

            if flag == 1 and b_sim_info != None:
                if i % 100 == 0:
                    print(f'(front: {start_buffer._p} / {buffer_size})')
                i = i + 1
                start_buffer.append(np.array([sim_info['state'][0]]), sim_info['state'][1], sim_info['state'][2], a, np.array([-1]))
                # print('append')
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
            # env.render()
            # time.sleep(1e-2)
            o = np.array(o, dtype=np.float32)

            a, vpred = policy.act(o, False)
            n_o, r, d, info = env.step(a)
            n_o = np.array(n_o, dtype=np.float32)

            if args.complex_task == 'pick':
                flag = env.unwrapped.is_boundary_pick_rear(o, n_o)
            elif args.complex_task == 'serve':
                flag = env.unwrapped.is_boundary_hit_rear()
            elif args.complex_task == 'catch':
                flag = env.unwrapped.is_boundary_catch_rear()

            if flag == 1:
                states.append(o)
                buffer.append(o, a, r, d, n_o)
                # print('append')
                if i % 1000 == 0:
                    print(f'(rear: {i}, {buffer_size})')
                i = i + 1
                episode_return += r

            o = n_o
            
            if d or flag == -1:
                # print('done')
                num_episodes += 1
                total_return += episode_return
                o = env.reset()
                episode_return = 0.0

            

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

