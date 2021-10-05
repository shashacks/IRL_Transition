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

def set_seed(seed, env):
    # Random seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    random.seed(seed)   
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)       # for multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    env.seed(seed)

def collect_demo(args):

    sess = U.single_threaded_session(gpu=False)
    sess.__enter__()

    env_name = args.env
    env = make_env(env_name)
    set_seed(args.seed, env)
    
    env.unwrapped.set_curbs_x_randomness_for_irl()

    set_global_seeds(args.seed)

        # build vanilla TRPO
    policy = MlpPolicy(
        env=env,
        name="%s/pi" % env_name,
        ob_env_name=env_name,
        config=args)

    old_policy = MlpPolicy(
        env=env,
        name="%s/oldpi" % env_name,
        ob_env_name=env_name,
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
    # if args.env == 'Walker2dJump-v1':
    #     obs_shape -= 2
    if args.env == 'Walker2dForward-v1' or args.env == 'Walker2dCrawl-v1':
        obs_shape += 2

    buffer_size = args.exp_buffer_size
    buffer = Buffer(
        buffer_size= buffer_size,
        state_shape=obs_shape,
        action_shape=act_shape,
        device=torch.device('cpu')
    )
    

    if args.env == 'Walker2dForward-v1' or args.env == 'Walker2dCrawl-v1' or args.env == 'Walker2dJump-v1':
        start_buffer = StartBuffer(
            buffer_size= buffer_size // 100,
            time=1,
            qpos=env.unwrapped.model.nq,
            qvel=env.unwrapped.model.nv,
            qact=act_shape,
            obstacle_pos=1,
            device=torch.device('cpu')
        )


    total_return = 0.0
    num_episodes = 0

    o = env.reset()
    if args.env == 'Walker2dForward-v1' or args.env == 'Walker2dCrawl-v1':
        dist = env.unwrapped.get_dist()
    episode_return = 0.0
    i = 0
    

    while i < buffer_size:
        sim_info = env.unwrapped.get_sim_state()
        a, vpred = policy.act(o, False)
        n_o, r, d, info = env.step(a)
        print(env.unwrapped.get_cur_pos(), env.unwrapped.get_x_pos())

        if args.env == 'Walker2dForward-v1' or args.env == 'Walker2dCrawl-v1':
            n_dist = env.unwrapped.get_dist()
        if args.front:
            flag = env.unwrapped.is_boundary()
        else:
            flag = env.unwrapped.is_boundary_rear()
        if flag == 0:
            i = i + 1
            if args.env == 'Walker2dForward-v1' or args.env == 'Walker2dCrawl-v1':
                buffer.append(np.append(o, dist), a, r, d, np.append(n_o, n_dist))
            else:
                buffer.append(o, a, r, d, n_o)
            env.render()
            time.sleep(5e-3)
            start_buffer.append(np.array([b_sim_info['state'][0]]), b_sim_info['state'][1], b_sim_info['state'][2], b_ac, np.array([b_sim_info['obstacle_pos']]))
            
            if i % 1000 == 0:
                print(f'({i}, {start_buffer._p}) / {buffer_size}')
                print(env.unwrapped.get_curbs_x_randomness())

            episode_return += r
        if d or flag == -1:
            num_episodes += 1
            total_return += episode_return
            o = env.reset()
            episode_return = 0.0
            

        o = n_o
        if args.env == 'Walker2dForward-v1' or args.env == 'Walker2dCrawl-v1':
            dist = n_dist
        b_sim_info = sim_info
        b_ac = a

    print(f'Mean return of the expert is {total_return / num_episodes}')
    
    buffer.save(os.path.join(
        'data',
        'exp_demo',
        args.env + '_' +  args.suffix,
        'exp_demo.pth'
    ))
    start_buffer.save(os.path.join(
        'data',
        'exp_demo',
        args.env + '_' +  args.suffix,
        'exp_demo_start.pth'
    ))
