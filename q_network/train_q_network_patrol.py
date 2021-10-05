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

def save_qnet(prev_updates, updates, fpath, q13, q31, q23, q32, save_frequency):
    
    if prev_updates[0] != updates[0]:
        if updates[0] % save_frequency == 0:
            fname = str(updates[0]) + '_q13.pt'
            q13.save_model(fpath, fname)

    if prev_updates[1] != updates[1]:
        if updates[1] % save_frequency == 0:
            fname = str(updates[1]) + '_q31.pt'
            q31.save_model(fpath, fname)

    if prev_updates[2] != updates[2]:
        if updates[2] % save_frequency == 0:
            fname = str(updates[2]) + '_q23.pt'
            q23.save_model(fpath, fname)

    if prev_updates[3] != updates[3]:
        if updates[3] % save_frequency == 0:
            fname = str(updates[3]) + '_q32.pt'
            q32.save_model(fpath, fname)

def act_random_q_net(obs, obs_next, d, pivot, q_random, arr, cur_pi, next_pi, q, duration, cur_duration):
    if cur_duration < pivot: # stay
        arr[1] += 1
        q.buffer.add(obs, 1, 0, obs_next, d)
        return False, pivot, q_random, cur_pi, obs_next
    elif pivot <= cur_duration:
        pivot = np.random.randint(duration) + 1 
        q_net_pi12_random = random.random()
        temp_obs_for_q_net_pi12 = obs_next # keep observation
        return True, pivot, q_net_pi12_random, next_pi, temp_obs_for_q_net_pi12

def act_non_random_q_net(obs, obs_next, d, pivot, q_net_random, arr, cur_pi, next_pi, q, max_update, batch_size, updates, losses, q_idx, duration, cur_duration):
    q_a = q.act(obs_next)
    if q_a == 0: # convertd
        pivot = np.random.randint(duration) + 1 
        success_fail_for_q_net = True
        temp_obs_for_q_net_pi12 = obs_next # keep observation
        return success_fail_for_q_net, pivot, q_net_random, next_pi, temp_obs_for_q_net_pi12
    elif q_a == 1: # stay
        if d:
            q_net_random = random.random()
            success_fail_for_q_net = False
            arr[2] += 1
            q.buffer.add(obs, 1, -1, obs_next, d) # fail reward
            if q.buffer.size() > batch_size and updates[q_idx] < max_update:
                loss = q.learning(updates[q_idx])
                losses[q_idx] = np.append(losses[q_idx], loss)
                updates[q_idx] += 1
            return success_fail_for_q_net, pivot, q_net_random, cur_pi, None
        elif duration < cur_duration:         
            pivot = np.random.randint(duration) + 1 
            q_net_random = random.random()
            success_fail_for_q_net = False
            arr[3] += 1
            q.buffer.add(obs, 1, -1, obs_next, d)
            if q.buffer.size() > batch_size and updates[q_idx] < max_update:
                loss = q.learning(updates[0])
                losses[q_idx] = np.append(losses[q_idx], loss)
                updates[q_idx] += 1
            return success_fail_for_q_net, pivot, q_net_random, next_pi, None
        else:
            arr[4] += 1
            q.buffer.add(obs, 1, 0, obs_next, d)
            return False, pivot, q_net_random, cur_pi, None

def train_q_network_patrol(args):

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
    eval_frequency = args.eval_frequency_q

    q13 = DQN_Converter(args, dim, batch_size)
    q31 = DQN_Converter(args, dim, batch_size)
    q23 = DQN_Converter(args, dim, batch_size)
    q32 = DQN_Converter(args, dim, batch_size)
    q13.set_mean_std(pi13.get_mean(), pi13.get_std())
    q31.set_mean_std(pi31.get_mean(), pi31.get_std())
    q23.set_mean_std(pi23.get_mean(), pi23.get_std())
    q32.set_mean_std(pi32.get_mean(), pi32.get_std())

    # non-Linear epsilon decay
    epsilon_final = args.epsilon_min_q
    epsilon_start = args.epsilon_q
    epsilon_decay = args.epsilon_decay_q
    epsilon_by_frame = lambda epoch_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(
        -1. * epoch_idx / epsilon_decay)


    fail_reward = -1.0
    success_reward = 1.0
    losses = [np.array([]), np.array([]), np.array([]), np.array([])]
    updates = [0, 0, 0, 0]
    prev_updates = [-1, -1, -1, -1]
    max_update = 30000

    tot_interaction = 0
    arr1 = [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0] 
    arr2 = [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0] 
    arr3 = [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0] 
    arr4 = [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0] 
    
    q_net_flags = [False, False, False, False]

    save_frequency = 1000
    fpath = osp.join('data', 'q_network', args.fname)
    os.makedirs(fpath, exist_ok=True)
    fname = osp.join(fpath, 'progress.txt')
    f = open(fname, 'w')

    obs = env.reset()
    obs = obs[:-1]
    q_net_pi13_random = random.random()
    q_net_pi31_random = random.random()
    q_net_pi23_random = random.random()
    q_net_pi32_random = random.random()
    success_fail_for_q_net_pi13 = False
    success_fail_for_q_net_pi31 = False
    success_fail_for_q_net_pi23 = False
    success_fail_for_q_net_pi32 = False
    temp_obs_for_q_net_pi13 = None
    temp_obs_for_q_net_pi31 = None
    temp_obs_for_q_net_pi23 = None
    temp_obs_for_q_net_pi32 = None

    duration = 100
    pivot = np.random.randint(duration) + 1
    policy = pi1 if env.unwrapped._direction == 1 else pi2
    success_count = 1
    cur_duration = 0
    success_count_list = []
    mean_max = 0
    while not (q_net_flags[0] and q_net_flags[1] and q_net_flags[2] and q_net_flags[3]):
        
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
            # print('cur pi13')
            cur_duration += 1
            a = policy.exploit(obs)
        elif policy == pi31:
            # print('cur pi31')
            cur_duration += 1
            a = policy.exploit(obs)
        elif policy == pi23:
            # print('cur pi23')
            cur_duration += 1
            a = policy.exploit(obs)
        elif policy == pi32:
            # print('cur pi32')
            cur_duration += 1
            a = policy.exploit(obs)

        obs_next, r, d, info = env.step(a)
        obs_next = obs_next[:-1]
        
        tot_interaction += 1
        # env.render()
        # time.sleep(1e-2)

        if policy == pi1 and env.unwrapped.is_terminate('walk'):
            policy = pi13
        elif policy == pi13 and q_net_pi13_random <= epsilon_by_frame(updates[0]): # random action
            success_fail_for_q_net_pi13, pivot, q_net_pi13_random, policy, temp_obs_for_q_net_pi13 = \
                act_random_q_net(obs, obs_next, d, pivot, q_net_pi13_random, arr1, policy, pi3, q13, duration, cur_duration)
            if policy == pi3:
                env.unwrapped.is_terminate('balance', init=True)
        elif policy == pi13 and q_net_pi13_random > epsilon_by_frame(updates[0]):
            success_fail_for_q_net_pi13, pivot, q_net_pi13_random, policy, temp_obs_for_q_net_pi13 = \
                act_non_random_q_net(obs, obs_next, d, pivot, q_net_pi13_random, arr1, policy, pi3, q13, max_update, batch_size, updates, losses, 0, duration, cur_duration)
            if policy == pi3:
                env.unwrapped.is_terminate('balance', init=True)
        elif policy == pi31 and q_net_pi31_random <= epsilon_by_frame(updates[1]): # random action
            success_fail_for_q_net_pi31, pivot, q_net_pi31_random, policy, temp_obs_for_q_net_pi31 = \
                act_random_q_net(obs, obs_next, d, pivot, q_net_pi31_random, arr2, policy, pi1, q31, duration, cur_duration)
        elif policy == pi31 and q_net_pi31_random > epsilon_by_frame(updates[1]):
            success_fail_for_q_net_pi31, pivot, q_net_pi31_random, policy, temp_obs_for_q_net_pi31 = \
                act_non_random_q_net(obs, obs_next, d, pivot, q_net_pi31_random, arr2, policy, pi1, q31, max_update, batch_size, updates, losses, 1, duration, cur_duration)
        
        elif policy == pi2 and env.unwrapped.is_terminate('walk'):
            policy = pi23
        elif policy == pi23 and q_net_pi23_random <= epsilon_by_frame(updates[2]): # random action
            success_fail_for_q_net_pi23, pivot, q_net_pi23_random, policy, temp_obs_for_q_net_pi23 = \
                act_random_q_net(obs, obs_next, d, pivot, q_net_pi23_random, arr3, policy, pi3, q23, duration, cur_duration)
            if policy == pi3:
                env.unwrapped.is_terminate('balance', init=True)
        elif policy == pi23 and q_net_pi23_random > epsilon_by_frame(updates[2]):
            success_fail_for_q_net_pi23, pivot, q_net_pi23_random, policy, temp_obs_for_q_net_pi23 = \
                act_non_random_q_net(obs, obs_next, d, pivot, q_net_pi31_random, arr3, policy, pi3, q23, max_update, batch_size, updates, losses, 1, duration, cur_duration)
            if policy == pi3:
                env.unwrapped.is_terminate('balance', init=True)
        elif policy == pi32 and q_net_pi32_random <= epsilon_by_frame(updates[3]): # random action
            success_fail_for_q_net_pi32, pivot, q_net_pi32_random, policy, temp_obs_for_q_net_pi32 = \
                act_random_q_net(obs, obs_next, d, pivot, q_net_pi32_random, arr4, policy, pi2, q32, duration, cur_duration)
        elif policy == pi32 and q_net_pi32_random > epsilon_by_frame(updates[3]):
            success_fail_for_q_net_pi32, pivot, q_net_pi32_random, policy, temp_obs_for_q_net_pi32 = \
                act_non_random_q_net(obs, obs_next, d, pivot, q_net_pi32_random, arr4, policy, pi2, q32, max_update, batch_size, updates, losses, 1, duration, cur_duration)
               
        elif policy == pi3 and env.unwrapped.is_terminate('balance'):
            x_pos = env.unwrapped.get_x_pos()
            if 1 < x_pos: 
                policy = pi32
            if -1 > x_pos:
                policy = pi31

        if success_fail_for_q_net_pi13 and policy == pi23:
            success_fail_for_q_net_pi13 = False
            arr1[5] += 1
            q13.buffer.add(temp_obs_for_q_net_pi13, 0, success_reward, obs_next, d) # successful execution
            if q13.buffer.size() > batch_size and updates[0] < max_update:
                loss = q13.learning(updates[0])
                losses[0] = np.append(losses[0], loss)
                updates[0] += 1
        elif success_fail_for_q_net_pi13 and d:
            success_fail_for_q_net_pi13 = False
            arr1[6] += 1
            q13.buffer.add(temp_obs_for_q_net_pi13, 0, fail_reward, obs_next, d) # unsuccessful execution
            if q13.buffer.size() > batch_size and updates[0] < max_update:
                loss = q13.learning(updates[0])
                losses[0] = np.append(losses[0], loss)
                updates[0] += 1
    
        if success_fail_for_q_net_pi31 and policy == pi13:
            success_fail_for_q_net_pi31 = False
            arr2[5] += 1
            q31.buffer.add(temp_obs_for_q_net_pi31, 0, success_reward, obs_next, d) # successful execution
            if q31.buffer.size() > batch_size and updates[1] < max_update:
                loss = q31.learning(updates[1])
                losses[1] = np.append(losses[1], loss)
                updates[1] += 1
        elif success_fail_for_q_net_pi31 and d:
            success_fail_for_q_net_pi31 = False
            arr2[6] += 1
            q31.buffer.add(temp_obs_for_q_net_pi31, 0, fail_reward, obs_next, d) # unsuccessful execution
            if q31.buffer.size() > batch_size and updates[1] < max_update:
                loss = q31.learning(updates[1])
                losses[1] = np.append(losses[1], loss)
                updates[1] += 1
        
        if success_fail_for_q_net_pi23 and policy == pi13:
            success_fail_for_q_net_pi23 = False
            arr3[5] += 1
            q23.buffer.add(temp_obs_for_q_net_pi23, 0, success_reward, obs_next, d) # successful execution
            if q23.buffer.size() > batch_size and updates[2] < max_update:
                loss = q23.learning(updates[2])
                losses[2] = np.append(losses[2], loss)
                updates[2] += 1
        elif success_fail_for_q_net_pi23 and d:
            success_fail_for_q_net_pi23 = False
            arr3[6] += 1
            q23.buffer.add(temp_obs_for_q_net_pi23, 0, fail_reward, obs_next, d) # unsuccessful execution
            if q23.buffer.size() > batch_size and updates[2] < max_update:
                loss = q23.learning(updates[2])
                losses[2] = np.append(losses[2], loss)
                updates[2] += 1

        if success_fail_for_q_net_pi32 and policy == pi23:
            success_fail_for_q_net_pi32 = False
            arr4[5] += 1
            q32.buffer.add(temp_obs_for_q_net_pi32, 0, success_reward, obs_next, d) # successful execution
            if q32.buffer.size() > batch_size and updates[3] < max_update:
                loss = q32.learning(updates[3])
                losses[3] = np.append(losses[3], loss)
                updates[3] += 1
        elif success_fail_for_q_net_pi32 and d:
            success_fail_for_q_net_pi32 = False
            arr4[6] += 1
            q32.buffer.add(temp_obs_for_q_net_pi32, 0, fail_reward, obs_next, d) # unsuccessful execution
            if q32.buffer.size() > batch_size and updates[3] < max_update:
                loss = q32.learning(updates[3])
                losses[3] = np.append(losses[3], loss)
                updates[3] += 1



        obs = obs_next

        if d:
            obs = env.reset()
            obs = obs[:-1]
            pivot = np.random.randint(duration) + 1
            policy = pi1 if env.unwrapped._direction == 1 else pi2
            cur_duration = 0
            if len(success_count_list) >= 100:
                success_count_list.pop(0)
            success_count_list.append(info['success_count'])
            if len(success_count_list) >= 100 and mean_max < np.array(success_count_list).mean():
                mean_max = np.array(success_count_list).mean()
                print('mean_max:', mean_max, updates)
                

        if updates[0] >= max_update:
            q_net_flags[0] = True
        if updates[1] >= max_update:
            q_net_flags[1] = True
        if updates[2] >= max_update:
            q_net_flags[2] = True
        if updates[3] >= max_update:
            q_net_flags[3] = True

        save_qnet(prev_updates, updates, fpath, q13, q31, q23, q32, save_frequency)

        for i in range(len(updates)):
            prev_updates[i] = updates[i]

        # if(sum(updates) % 2000 == 0):
        #     print(updates)
        #     print(arr1)
        #     print(arr2)
        #     print(arr3)
        #     print(arr4)

    

    print(arr1)
    print(arr2)
    print(arr3)
    print(arr4)
    print('total interaction:', tot_interaction)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    args = p.parse_args()
    train_q_network_patrol(args)

