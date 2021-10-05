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

def evaluate_network_random_without_tran(env_test, pi1):
    n = 0
    tot_iters = 50
    tot_success = 0
    policy = pi1

    success_res = np.array([])

    obs = env_test.reset()
    while n < tot_iters:
        # print('cur pi1')
        # env_test.render()
        # time.sleep(1e-2)
        a, vpred = policy.act(obs, False)
        obs, r, d, info = env_test.step(a)

        if d:
            tot_success += info['success_count']
            success_res = np.append(success_res, info['success_count'])
            obs = env_test.reset()
            print('done')
            n += 1

    print(f'without transition policy {tot_success} / {tot_iters}, ratio: {tot_success/tot_iters}')
    return success_res

def evaluate_network_random_without_qnet(env_test, pi1, pi12, duration):
    
    n = 0
    tot_iters = 50
    tot_success = 0
    
    policy = pi1
    pivot = np.random.randint(duration) + 1
    cur_duration = 0

    success_res = np.array([])

    obs = env_test.reset()
    while n < tot_iters:
        # env_test.render()
        # time.sleep(1e-2)
        if policy == pi1:
            print('cur pi1')
            a, vpred = policy.act(obs, False)
        elif policy == pi12:
            print('cur pi12')
            print(cur_duration, pivot)
            a = policy.exploit(obs)
            cur_duration += 1

        obs, r, d, info = env_test.step(a)

        if env_test.unwrapped.is_terminate():
            policy = pi12
            cur_duration = 0
        if policy == pi12 and cur_duration > pivot:
            pivot = np.random.randint(duration) + 1
            cur_duration = 0
            policy = pi1

        if d:
            tot_success += info['success_count']
            success_res = np.append(success_res, info['success_count'])
            obs = env_test.reset()
            cur_duration = 0
            policy = pi1
            pivot = np.random.randint(duration) + 1
            print('done')
            n += 1

    print(f'only pi1 -> pi12 -> pi2: tot_success: {tot_success}, ratio: {tot_success / tot_iters}')
    return success_res

def evaluate_network_random_with_tran_and_qnet(env_test, pi1, pi12, q12, duration):
    
    n = 0
    tot_iters = 50
    tot_success = 0
    
    policy = pi1
    cur_duration = 0
    force_1_to_2 = 0
    success_res = np.array([])
    obs = env_test.reset()
    while n < tot_iters:
    #     env_test.render()
    #     time.sleep(1e-2)
        if policy == pi1:
            print('cur pi1')
            a, vpred = policy.act(obs, False)
            cur_duration = 0
        elif policy == pi12:
            print('cur pi12')
            a = policy.exploit(obs)
            cur_duration += 1

        obs, r, d, info = env_test.step(a)

        if env_test.unwrapped.is_terminate():
            policy = pi12
            cur_duration = 0
        elif policy == pi12:
            guess = q12.act(obs)
            if guess == 0:
                policy = pi1
                print('change')
            elif 30 < cur_duration:
                policy = pi1
                force_1_to_2 += 1

        if d:
            tot_success += info['success_count']
            success_res = np.append(success_res, info['success_count'])
            obs = env_test.reset()
            cur_duration = 0
            policy = pi1
            print('done')
            n += 1


    print(f'with Q-network pi1 -> pi12 -> pi2: tot_success: {tot_success}, ratio: {tot_success / tot_iters}')
    print('force_1_to_2: ', force_1_to_2)
    return success_res


def test_q_network_pick(args):
    print('test pick')
    sess = U.single_threaded_session(gpu=False)
    sess.__enter__()

    env = make_env(args.env)
    env_test = make_env(args.env)
    pi1_env = make_env(args.pi1_env)
    pi1_env.seed(args.seed)
    
    set_seed(args.seed, env, env_test)
    set_global_seeds(args.seed)

    pi1 = load_primitive_policy(pi1_env, args.pi1_env, args.pi1, args)
    pi12 = PPOExpert(
        state_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        device=torch.device("cpu"),
        path=args.pi12)
    print(env.observation_space.shape)
    exit()
    dim = 1
    for d in env.observation_space.shape:
        dim = dim * d

    batch_size = 64 #args.batch_size_q
    eval_frequency = args.eval_frequency_q

    q12 = DQN_Converter(args, dim, batch_size)
    q12.load_weights(args.q12)
    duration = 40
    success_res = evaluate_network_random_without_tran(env_test, pi1)
    # success_res = evaluate_network_random_without_qnet(env_test, pi1, pi12, duration)
    # success_res = evaluate_network_random_with_tran_and_qnet(env_test, pi1, pi12, q12, duration)
    for v in success_res:
        print(f'{v}, ', end='')
    # print('end')

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    args = p.parse_args()
    test_q_network_pick(args)


