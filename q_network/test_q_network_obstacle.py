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
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)  
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    tf.random.set_random_seed(seed)
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

def evaluate_network_random_without_tran(env_test, pi1, pi2, pi3, const):
    obs = env_test.reset()

    n = 0
    tot_iters = 50
    tot_success = 0
    
    intervals = []
    obstacle = env_test.unwrapped.get_obstacle_pos_and_type()
    print(obstacle['type'])
    order = []
    success_res = np.array([])
    for i in range(len(obstacle['pos'])):
        offset_b = 3.1 if obstacle['type'][i] else 4.5
        offset_a = 2.5 if obstacle['type'][i] else 2.0
        intervals.append([obstacle['pos'][i] - offset_b, obstacle['pos'][i] - offset_b + const])
        intervals.append([obstacle['pos'][i] + offset_a, obstacle['pos'][i] + offset_a + const])
        if obstacle['type'][i]:
            order.append('pi12')
            order.append('pi21')
        else:
            order.append('pi13')
            order.append('pi31')

    idx = 0
    pivot = intervals[idx][0] + const * np.random.rand(1)
    policy = pi1

    while n < tot_iters:
        if policy == pi1:
            print('cur pi1')
            a, vpred = policy.act(obs[:-2], False)
        elif policy == pi2:
            print('cur pi2')
            a, vpred = policy.act(obs, False)
        elif policy == pi3:
            print('cur pi3')
            a, vpred = policy.act(obs[:-2], False)

        obs, r, d, info = env_test.step(a)
        x_pos = env_test.unwrapped.get_x_pos()
        # print(order, idx)
        # print(env_test.unwrapped.get_obstacle_pos_and_type()['pos'])
        # print(intervals)
        
        # env_test.render()
        # time.sleep(5e-3)
        print(n, x_pos)
        if policy == pi1 and pivot < x_pos and order[idx] == 'pi12':
            policy = pi2
            # print('pi2')
            idx += 1
            pivot = intervals[idx][0] + const * np.random.rand(1)
        elif policy == pi2 and pivot < x_pos:
            policy = pi1
            # print('pi1')
            idx += 1
            pivot = intervals[idx][0] + const * np.random.rand(1)
        elif policy == pi1 and pivot < x_pos and order[idx] == 'pi13':
            policy = pi3
            # print('pi3')
            idx += 1
            pivot = intervals[idx][0] + const * np.random.rand(1)
        elif policy == pi3 and pivot < x_pos:
            policy = pi1
            # print('pi1')
            idx += 1
            pivot = intervals[idx][0] + const * np.random.rand(1)


        if d:
            tot_success += info['success_count']
            success_res = np.append(success_res, info['success_count'])
            if n % 100 == 0:
                print(f'{n}')
            n += 1
            obs = env_test.reset()
            intervals = []
            obstacle = env_test.unwrapped.get_obstacle_pos_and_type()
            order = []
            for i in range(len(obstacle['pos'])):
                offset_b = 3.1 if obstacle['type'][i] else 4.5
                offset_a = 2.5 if obstacle['type'][i] else 2.0
                intervals.append([obstacle['pos'][i] - offset_b, obstacle['pos'][i] - offset_b + const])
                intervals.append([obstacle['pos'][i] + offset_a, obstacle['pos'][i] + offset_a + const])
                if obstacle['type'][i]:
                    order.append('pi12')
                    order.append('pi21')
                else:
                    order.append('pi13')
                    order.append('pi31')
            idx = 0
            pivot = intervals[idx][0] + const * np.random.rand(1)
            policy = pi1

            
    print(f'without transition policy {tot_success} / {tot_iters}, ratio: {tot_success/tot_iters}')
    return success_res

def evaluate_network_random_without_qnet(env_test, pi1, pi2, pi3, pi12, pi21, pi13, pi31, const):
    
    obs = env_test.reset()

    n = 0
    tot_iters = 50
    tot_success = 0
    
    intervals = []
    obstacle = env_test.unwrapped.get_obstacle_pos_and_type()
    print(obstacle['type'])
    order = []
    success_res = np.array([])
    for i in range(len(obstacle['pos'])):
        offset_b = 3.1 if obstacle['type'][i] else 4.5
        offset_a = 2.5 if obstacle['type'][i] else 2.0
        intervals.append([obstacle['pos'][i] - offset_b, obstacle['pos'][i] - offset_b + const])
        intervals.append([obstacle['pos'][i] + offset_a, obstacle['pos'][i] + offset_a + const])
        if obstacle['type'][i]:
            order.append('pi12')
            order.append('pi21')
        else:
            order.append('pi13')
            order.append('pi31')

    idx = 0
    pivot = intervals[idx][0] + const * np.random.rand(1)
    policy = pi1

    while n < tot_iters:
        if policy == pi1:
            print('cur pi1')
            a, vpred = policy.act(obs[:-2], False)
        elif policy == pi2:
            print('cur pi2')
            a, vpred = policy.act(obs, False)
        elif policy == pi3:
            print('cur pi3')
            a, vpred = policy.act(obs[:-2], False)
        elif policy == pi12:
            print('cur pi12')
            a = policy.exploit(obs)
        elif policy == pi21:
            print('cur pi21')
            a = policy.exploit(obs)
        elif policy == pi13:
            print('cur pi13')
            a = policy.exploit(obs)
        elif policy == pi31:
            print('cur pi31')
            a = policy.exploit(obs)

        obs, r, d, info = env_test.step(a)
        x_pos = env_test.unwrapped.get_x_pos()
        # print(order, idx)
        # print(env_test.unwrapped.get_obstacle_pos_and_type()['pos'])
        # print(intervals)
        
        # env_test.render()
        # time.sleep(5e-3)
        print(n, x_pos)
        if policy == pi1 and intervals[idx][0] <= x_pos and order[idx] == 'pi12':
            policy = pi12
            # print('pi12')
        elif policy == pi12 and pivot < x_pos:
            policy = pi2
            # print('pi2')
            idx += 1
            pivot = intervals[idx][0] + const * np.random.rand(1)
        elif policy == pi2 and intervals[idx][0] <= x_pos:
            policy = pi21
            # print('pi21')
        elif policy == pi21 and pivot < x_pos:
            policy = pi1
            # print('pi1')
            idx += 1
            pivot = intervals[idx][0] + const * np.random.rand(1)
        elif policy == pi1 and intervals[idx][0] <= x_pos and order[idx] == 'pi13':
            policy = pi13
            # print('pi13')
        elif policy == pi13 and pivot < x_pos:
            policy = pi3
            # print('pi3')
            idx += 1
            pivot = intervals[idx][0] + const * np.random.rand(1)
        elif policy == pi3 and intervals[idx][0] <= x_pos:
            policy = pi31
            # print('pi31')
        elif policy == pi31 and pivot < x_pos:
            policy = pi1
            # print('pi1')
            idx += 1
            pivot = intervals[idx][0] + const * np.random.rand(1)


        if d:
            tot_success += info['success_count']
            success_res = np.append(success_res, info['success_count'])
            if n % 100 == 0:
                print(f'{n}')
            n += 1
            obs = env_test.reset()
            intervals = []
            obstacle = env_test.unwrapped.get_obstacle_pos_and_type()
            order = []
            for i in range(len(obstacle['pos'])):
                offset_b = 3.1 if obstacle['type'][i] else 4.5
                offset_a = 2.5 if obstacle['type'][i] else 2.0
                intervals.append([obstacle['pos'][i] - offset_b, obstacle['pos'][i] - offset_b + const])
                intervals.append([obstacle['pos'][i] + offset_a, obstacle['pos'][i] + offset_a + const])
                if obstacle['type'][i]:
                    order.append('pi12')
                    order.append('pi21')
                else:
                    order.append('pi13')
                    order.append('pi31')
            idx = 0
            pivot = intervals[idx][0] + const * np.random.rand(1)
            policy = pi1

            
    print(f'with transition policy {tot_success} / {tot_iters}, ratio: {tot_success/tot_iters}')
    return success_res

def evaluate_network_random_with_tran_and_qnet(env_test, pi1, pi2, pi3, pi12, pi21, pi13, pi31, q12, q21, q13, q31, const):

    n = 0
    tot_iters = 50
    tot_success = 0
    force_1_to_2 = 0
    force_2_to_1 = 0
    force_1_to_3 = 0
    force_3_to_1 = 0
    
    policy = pi1

    intervals = []
    obstacle = env_test.unwrapped.get_obstacle_pos_and_type()
    print(obstacle['type'])
    order = []
    success_res = np.array([])
    for i in range(len(obstacle['pos'])):
        offset_b = 3.1 if obstacle['type'][i] else 4.5
        offset_a = 2.5 if obstacle['type'][i] else 2.0
        intervals.append([obstacle['pos'][i] - offset_b, obstacle['pos'][i] - offset_b + const])
        intervals.append([obstacle['pos'][i] + offset_a, obstacle['pos'][i] + offset_a + const])
        if obstacle['type'][i]:
            order.append('pi12')
            order.append('pi21')
        else:
            order.append('pi13')
            order.append('pi31')

    obs = env_test.reset()
    idx  = 0

    while n < tot_iters:
        # if n == 2:
        #     env_test.render()
        #     time.sleep(1e-2)
        if policy == pi1:
            # print('cur pi1')
            a, vpred = policy.act(obs[:-2], False)
        elif policy == pi2:
            # print('cur pi2')
            a, vpred = policy.act(obs, False)
        elif policy == pi3:
            # print('cur pi3')
            a, vpred = policy.act(obs[:-2], False)
        elif policy == pi12:
            # print('cur pi12')
            a = policy.exploit(obs)
        elif policy == pi21:
            # print('cur pi21')
            a = policy.exploit(obs)
        elif policy == pi13:
            # print('cur pi13')
            a = policy.exploit(obs)
        elif policy == pi31:
            # print('cur pi31')
            a = policy.exploit(obs)

        # env_test.render()
        obs, r, d, info = env_test.step(a)
        x_pos = env_test.unwrapped.get_x_pos()
        print(n, x_pos)
        

        if policy == pi1 and intervals[idx][0] <= x_pos:
            if order[idx] == 'pi12':
                policy = pi12 
            elif order[idx] == 'pi13':
                policy = pi13
        elif policy == pi12 and intervals[idx][0] <= x_pos and x_pos <= intervals[idx][1]:
            guess = q12.act(obs)
            if guess == 0:
                policy = pi2
                idx += 1
        elif policy == pi12 and intervals[idx][1] < x_pos:
            policy = pi2
            idx += 1
            force_1_to_2 += 1
        elif policy == pi13 and intervals[idx][0] <= x_pos and x_pos <= intervals[idx][1]:
            guess = q13.act(obs)
            if guess == 0:
                policy = pi3
                idx += 1
        elif policy == pi13 and intervals[idx][1] < x_pos:
            policy = pi3
            idx += 1
            force_1_to_3 += 1

        elif policy == pi2 and intervals[idx][0] <= x_pos:
            policy = pi21
        elif policy == pi21 and intervals[idx][0] <= x_pos and x_pos <= intervals[idx][1]:
            guess = q21.act(obs)
            if guess == 0:
                policy = pi1
                idx += 1
        elif policy == pi21  and intervals[idx][1] < x_pos:
            policy = pi1
            idx += 1
            force_2_to_1 += 1

        elif policy == pi3 and intervals[idx][0] <= x_pos:
            policy = pi31
        elif policy == pi31 and intervals[idx][0] <= x_pos and x_pos <= intervals[idx][1]:
            guess = q31.act(obs)
            if guess == 0:
                policy = pi1
                idx += 1
        elif policy == pi31  and intervals[idx][1] < x_pos:
            policy = pi1
            idx += 1
            force_3_to_1 += 1

        if d:
            tot_success += info['success_count']
            success_res = np.append(success_res, info['success_count'])
            # if info['success_count'] == 5:
            #     print(n)
            #     exit()
            n += 1
            obs = env_test.reset()
            intervals = []
            obstacle = env_test.unwrapped.get_obstacle_pos_and_type()
            print(obstacle['type'])
            order = []
            for i in range(len(obstacle['pos'])):
                offset_b = 3.1 if obstacle['type'][i] else 4.5
                offset_a = 2.5 if obstacle['type'][i] else 2.0
                intervals.append([obstacle['pos'][i] - offset_b, obstacle['pos'][i] - offset_b + const])
                intervals.append([obstacle['pos'][i] + offset_a, obstacle['pos'][i] + offset_a + const])
                if obstacle['type'][i]:
                    order.append('pi12')
                    order.append('pi21')
                else:
                    order.append('pi13')
                    order.append('pi31')
            policy = pi1
            idx = 0
            
    print(f'with transition policy and Q-net {tot_success} / {tot_iters}, ratio: {tot_success/tot_iters}')
    print(f'force_1_to_2: {force_1_to_2}, force_2_to_1: {force_2_to_1}, force_1_to_3: {force_1_to_3}, force_3_to_1: {force_3_to_1}')
    return success_res

def test_q_network_obstacle(args):

    print('test obstacle')
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
    
    
    pi12 = PPOExpert(
        state_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        device=torch.device("cpu"),
        path=args.pi12)

    pi21 = PPOExpert(
        state_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        device=torch.device("cpu"),
        path=args.pi21)

    pi13 = PPOExpert(
        state_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        device=torch.device("cpu"),
        path=args.pi13)

    pi31 = PPOExpert(
        state_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        device=torch.device("cpu"),
        path=args.pi31)


    dim = 1
    for d in env.observation_space.shape:
        dim = dim * d

    batch_size = 64 #args.batch_size_q
    eval_frequency = args.eval_frequency_q

    q12 = DQN_Converter(args, dim, batch_size)
    q21 = DQN_Converter(args, dim, batch_size)
    q13 = DQN_Converter(args, dim, batch_size)
    q31 = DQN_Converter(args, dim, batch_size)
    q12.load_weights(args.q12)
    q21.load_weights(args.q21)
    q13.load_weights(args.q13)
    q31.load_weights(args.q31)

    const = 1.0
    # success_res = evaluate_network_random_without_tran(env_test, pi1, pi2, pi3, const)
    # success_res = evaluate_network_random_without_qnet(env_test, pi1, pi2, pi3, pi12, pi21, pi13, pi31, const)
    success_res = evaluate_network_random_with_tran_and_qnet(env_test, pi1, pi2, pi3, pi12, pi21, pi13, pi31, q12, q21, q13, q31, const)
    for v in success_res:
        print(f'{v}, ', end='')
    print('done')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    args = p.parse_args()
    test_q_network_obstacle(args)

