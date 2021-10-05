import os
import os.path as osp
import argparse
import time
import math
from datetime import datetime

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random
import numpy as np

from q_network.q_network import DQN_Converter
from irl.algo.ppo import PPOExpert

import tensorflow as tf
import h5py

import baselines.common.tf_util as U
from baselines.common import set_global_seeds
from baselines import logger
from baselines.common.atari_wrappers import TransitionEnvWrapper
from rl.util import make_env
from rl.mlp_policy import MlpPolicy

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

def set_seed(seed, env, env_test):
    # Random seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    random.seed(seed)  
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    env.seed(seed)
    env_test.seed(2**31-seed)

def load_primitive_policy(env, env_name, path, args):
    # build vanilla TRPO
    pi = MlpPolicy(
        env=env,
        name="%s/pi" % env_name,
        ob_env_name=env_name,
        config=args)

    pi_old = MlpPolicy(
        env=env,
        name="%s/oldpi" % env_name,
        ob_env_name=env_name,
        config=args)
    
    networks = []
    networks.append(pi)
    networks.append(pi_old)

    var_list = []
    for network in networks:
        var_list += network.get_variables()

    if True:
        for var in var_list:
            logger.info('{} {}'.format(var.name, tensor_description(var)))
    ckpt_path = load_model(path, var_list)

    return pi

def evaluate_network_random_without_tran(env_test, pi1, pi2, pi3, duration):
    n = 0
    tot_iters = 50
    tot_success = 0

    success_res = np.array([])

    obs = env_test.reset()
    obs = obs[:-1]
    policy = pi1 if env_test.unwrapped._direction == 1 else pi2
    while n < tot_iters: 
        # env_test.render()
        # time.sleep(1e-2)
        if policy == pi1:
            print('cur pi1')
            a, vpred = policy.act(obs, False)
        elif policy == pi2:
            print('cur pi2')
            a, vpred = policy.act(obs, False)
        elif policy == pi3:
            print('cur pi3')
            a, vpred = policy.act(obs, False)
        obs, r, d, info = env_test.step(a)
        obs = obs[:-1]

        if policy == pi1 and env_test.unwrapped.is_terminate('walk'):
            policy = pi3
            env_test.unwrapped.is_terminate('balance', init=True)
        elif policy == pi2 and env_test.unwrapped.is_terminate('walk'):
            policy = pi3
            env_test.unwrapped.is_terminate('balance', init=True)
        elif policy == pi3 and env_test.unwrapped.is_terminate('balance'):
            x_pos = env_test.unwrapped.get_x_pos()
            if 1 < x_pos: 
                policy = pi2
            if -1 > x_pos:
                policy = pi1

        if d:
            tot_success += info['success_count']
            success_res = np.append(success_res, info['success_count'])
            obs = env_test.reset()
            obs = obs[:-1]
            print('done')
            n += 1
            policy = pi1 if env_test.unwrapped._direction == 1 else pi2


    print(f'without transition policy {tot_success} / {tot_iters}, ratio: {tot_success/tot_iters}')
    return success_res

def evaluate_network_random_without_qnet(env_test, pi1, pi2, pi3, pi13, pi31, pi23, pi32, duration):
    n = 0
    tot_iters = 50
    tot_success = 0

    success_res = np.array([])

    obs = env_test.reset()
    obs = obs[:-1]
    policy = pi1 if env_test.unwrapped._direction == 1 else pi2
    pivot = np.random.randint(duration) + 1 
    cur_duration = 0
    while n < tot_iters: 
        # env_test.render()
        # time.sleep(1e-2)
        if policy == pi1:
            # print('cur pi1')
            cur_duration = 0
            a, vpred = policy.act(obs, False)
        elif policy == pi2:
            # print('cur pi2')
            cur_duration = 0
            a, vpred = policy.act(obs, False)
        elif policy == pi3:
            # print('cur pi3')
            cur_duration = 0
            a, vpred = policy.act(obs, False)
        elif policy == pi13:
            # print('cur pi13', cur_duration, pivot)
            cur_duration += 1
            a = policy.exploit(obs)
        elif policy == pi31:
            # print('cur pi31', cur_duration, pivot)
            cur_duration += 1
            a = policy.exploit(obs)
        elif policy == pi23:
            # print('cur pi23', cur_duration, pivot)
            cur_duration += 1
            a = policy.exploit(obs)
        elif policy == pi32:
            # print('cur pi32', cur_duration, pivot)
            cur_duration += 1
            a = policy.exploit(obs)
        obs, r, d, info = env_test.step(a)
        obs = obs[:-1]

        if policy == pi1 and env_test.unwrapped.is_terminate('walk'):
            policy = pi13
            env_test.unwrapped.is_terminate('balance', init=True)
        elif policy == pi2 and env_test.unwrapped.is_terminate('walk'):
            policy = pi23
            env_test.unwrapped.is_terminate('balance', init=True)
        elif policy == pi3 and env_test.unwrapped.is_terminate('balance'):
            x_pos = env_test.unwrapped.get_x_pos()
            if 1 < x_pos: 
                policy = pi32
            if -1 > x_pos:
                policy = pi31
        elif policy == pi13 and pivot < cur_duration:
            policy = pi3
            pivot = np.random.randint(duration) + 1
        elif policy == pi31 and pivot < cur_duration:
            policy = pi1
            pivot = np.random.randint(duration) + 1
        elif policy == pi23 and pivot < cur_duration:
            policy = pi3
            pivot = np.random.randint(duration) + 1
        elif policy == pi32 and pivot < cur_duration:
            policy = pi2
            pivot = np.random.randint(duration) + 1

        if d:
            if policy == pi1:
                print(n, 'die cur pi1')
            elif policy == pi2:
                print(n, 'die cur pi2')
            elif policy == pi3:
                print(n, 'die cur pi3')
            elif policy == pi13:
                print(n, 'die cur pi13', cur_duration, pivot)
            elif policy == pi31:
                print(n, 'die cur pi31', cur_duration, pivot)
            elif policy == pi23:
                print(n, 'die cur pi23', cur_duration, pivot)
            elif policy == pi32:
                print(n, 'die cur pi32', cur_duration, pivot)
            tot_success += info['success_count']
            success_res = np.append(success_res, info['success_count'])
            obs = env_test.reset()
            obs = obs[:-1]
            print('done')
            n += 1
            policy = pi1 if env_test.unwrapped._direction == 1 else pi2
            pivot = np.random.randint(duration) + 1 
            cur_duration = 0




    print(f'only pi1 -> pi12 -> pi2: tot_success: {tot_success}, ratio: {tot_success / tot_iters}')
    return success_res

def evaluate_network_random_with_tran_and_qnet(env_test, pi1, pi2, pi3, pi13, pi31, pi23, pi32, q13, q31, q23, q32, duration):
    
    n = 0
    tot_iters = 50
    tot_success = 0

    success_res = np.array([])

    obs = env_test.reset()
    obs = obs[:-1]
    policy = pi1 if env_test.unwrapped._direction == 1 else pi2
    cur_duration = 0
    force_1_to_3 = 0
    force_3_to_1 = 0
    force_2_to_3 = 0
    force_3_to_2 = 0
    
    while n < tot_iters: 
        # env_test.render()
        # time.sleep(1e-2)
        # print(env_test.unwrapped.get_x_pos(), env_test.unwrapped.get_q_pos())
        if policy == pi1:
            # print('cur pi1')
            cur_duration = 0
            a, vpred = policy.act(obs, False)
        elif policy == pi2:
            # print('cur pi2')
            cur_duration = 0
            a, vpred = policy.act(obs, False)
        elif policy == pi3:
            # print('cur pi3')
            cur_duration = 0
            a, vpred = policy.act(obs, False)
        elif policy == pi13:
            # print('cur pi13', cur_duration)
            cur_duration += 1
            a = policy.exploit(obs)
        elif policy == pi31:
            # print('cur pi31', cur_duration)
            cur_duration += 1
            a = policy.exploit(obs)
        elif policy == pi23:
            # print('cur pi23', cur_duration)
            cur_duration += 1
            a = policy.exploit(obs)
        elif policy == pi32:
            # print('cur pi32', cur_duration)
            cur_duration += 1
            a = policy.exploit(obs)
        obs, r, d, info = env_test.step(a)
        obs = obs[:-1]

        if policy == pi1 and env_test.unwrapped.is_terminate('walk'):
            policy = pi13
            env_test.unwrapped.is_terminate('balance', init=True)
        elif policy == pi2 and env_test.unwrapped.is_terminate('walk'):
            policy = pi23
            env_test.unwrapped.is_terminate('balance', init=True)
        elif policy == pi3 and env_test.unwrapped.is_terminate('balance'):
            x_pos = env_test.unwrapped.get_x_pos()
            if 1 < x_pos: 
                policy = pi32
            if -1 > x_pos:
                policy = pi31
        elif policy == pi13:
            guess = q13.act(obs)
            if guess == 0:
                policy = pi3
                # print('cur pi13', cur_duration)
            elif cur_duration >= duration:
                force_1_to_3 += 1
                policy = pi3
        elif policy == pi31:
            guess = q31.act(obs)
            if guess == 0:
                policy = pi1
                # print('cur pi31', cur_duration)
            elif cur_duration >= duration:
                force_3_to_1 += 1
                policy = pi1
        elif policy == pi23:
            guess = q23.act(obs)
            if guess == 0:
                policy = pi3
                # print('cur pi23', cur_duration)
            elif cur_duration >= duration:
                force_2_to_3 += 1
                policy = pi3
        elif policy == pi32:
            guess = q32.act(obs)
            if guess == 0:
                policy = pi2
                # print('cur pi32', cur_duration)
            elif cur_duration >= duration:
                force_3_to_2 += 1
                policy = pi2

        if d:
            if policy == pi1:
                print(n, 'die cur pi1')
            elif policy == pi2:
                print(n, 'die cur pi2')
            elif policy == pi3:
                print(n, 'die cur pi3')
            elif policy == pi13:
                print(n, 'die cur pi13', cur_duration)
            elif policy == pi31:
                print(n, 'die cur pi31', cur_duration)
            elif policy == pi23:
                print(n, 'die cur pi23', cur_duration)
            elif policy == pi32:
                print(n, 'die cur pi32', cur_duration)
            tot_success += info['success_count']
            success_res = np.append(success_res, info['success_count'])
            obs = env_test.reset()
            obs = obs[:-1]
            print('done')
            n += 1
            policy = pi1 if env_test.unwrapped._direction == 1 else pi2
            cur_duration = 0

    print(f'with Q-network pi1 -> pi12 -> pi2: tot_success: {tot_success}, ratio: {tot_success / tot_iters}')
    print('force: ', force_1_to_3, force_3_to_1, force_2_to_3, force_3_to_2)
    return success_res


def test_q_network_patrol(args):
    print('test patrol')

    sess = U.single_threaded_session(gpu=False)
    sess.__enter__()

    env = make_env(args.env)
    env_test = make_env(args.env)
    pi1_env = make_env(args.pi1_env)
    pi2_env = make_env(args.pi2_env)
    pi3_env = make_env(args.pi3_env)
    pi1_env.seed(args.seed)
    pi2_env.seed(args.seed)
    pi3_env.seed(args.seed)
    
    set_seed(args.seed, env, env_test)
    set_global_seeds(args.seed)

    pi1 = load_primitive_policy(pi1_env, args.pi1_env, args.pi1, args)
    pi2 = load_primitive_policy(pi2_env, args.pi2_env, args.pi2, args)
    pi3 = load_primitive_policy(pi3_env, args.pi3_env, args.pi3, args)
    
    pi13 = PPOExpert(
        state_shape=pi1_env.observation_space.shape,
        action_shape=pi1_env.action_space.shape,
        device=torch.device("cpu"),
        path=args.pi13)

    pi31 = PPOExpert(
        state_shape=pi1_env.observation_space.shape,
        action_shape=pi1_env.action_space.shape,
        device=torch.device("cpu"),
        path=args.pi31)

    pi23 = PPOExpert(
        state_shape=pi1_env.observation_space.shape,
        action_shape=pi1_env.action_space.shape,
        device=torch.device("cpu"),
        path=args.pi23)

    pi32 = PPOExpert(
        state_shape=pi1_env.observation_space.shape,
        action_shape=pi1_env.action_space.shape,
        device=torch.device("cpu"),
        path=args.pi32)

    dim = 1
    for d in env.observation_space.shape:
        dim = dim * d
    dim -= 1

    batch_size = 64 #args.batch_size_q

    q13 = DQN_Converter(args, dim, batch_size)
    q31 = DQN_Converter(args, dim, batch_size)
    q23 = DQN_Converter(args, dim, batch_size)
    q32 = DQN_Converter(args, dim, batch_size)
    q13.load_weights(args.q13)
    q31.load_weights(args.q31)
    q23.load_weights(args.q23)
    q32.load_weights(args.q32)

    duration = 100

    success_res = evaluate_network_random_without_tran(env_test, pi1, pi2, pi3, duration)
    # success_res = evaluate_network_random_without_qnet(env_test, pi1, pi2, pi3, pi13, pi31, pi23, pi32, duration)
    # success_res = evaluate_network_random_with_tran_and_qnet(env_test, pi1, pi2, pi3, pi13, pi31, pi23, pi32, q13, q31, q23, q32, duration)
    for v in success_res:
        print(f'{v}, ', end='')
    # print('end')

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    args = p.parse_args()
    test_q_network_patrol(args)


