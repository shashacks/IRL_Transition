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

def save_qnet(prev_updates, updates, fpath, q12, q21, q13, q31, save_frequency):
    
    if prev_updates[0] != updates[0]:
        if updates[0] % save_frequency == 0:
            fname = str(updates[0]) + '_q12.pt'
            q12.save_model(fpath, fname)

    if prev_updates[1] != updates[1]:
        if updates[1] % save_frequency == 0:
            fname = str(updates[1]) + '_q21.pt'
            q21.save_model(fpath, fname)

    if prev_updates[2] != updates[2]:
        if updates[2] % save_frequency == 0:
            fname = str(updates[2]) + '_q13.pt'
            q13.save_model(fpath, fname)

    if prev_updates[3] != updates[3]:
        if updates[3] % save_frequency == 0:
            fname = str(updates[3]) + '_q31.pt'
            q31.save_model(fpath, fname)

def act_random_q_net(x_pos, obs, obs_next, d, idx, intervals, pivot, q_random, arr, cur_pi, next_pi, q):
    if x_pos < pivot: # stay
        arr[1] += 1
        q.buffer.add(obs, 1, 0, obs_next, d)
        return idx, False, pivot, q_random, cur_pi, obs_next
    elif pivot <= x_pos:
        idx += 1
        pivot = intervals[idx][0] + np.random.rand(1) 
        q_net_pi12_random = random.random()
        temp_obs_for_q_net_pi12 = obs_next # keep observation
        return idx, True, pivot, q_net_pi12_random, next_pi, temp_obs_for_q_net_pi12

def act_non_random_q_net(x_pos, obs, obs_next, d, idx, intervals, pivot, q_net_random, arr, cur_pi, next_pi, q, max_update, batch_size, updates, losses, q_idx):
    q_a = q.act(obs_next)
    if q_a == 0: # convertd
        idx += 1 
        pivot = intervals[idx][0] + np.random.rand(1)
        success_fail_for_q_net = True
        temp_obs_for_q_net_pi12 = obs_next # keep observation
        return idx, success_fail_for_q_net, pivot, q_net_random, next_pi, temp_obs_for_q_net_pi12
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
            return idx, success_fail_for_q_net, pivot, q_net_random, cur_pi, None
        elif intervals[idx][1] < x_pos:         
            idx += 1
            pivot = intervals[idx][0] + np.random.rand(1)   
            q_net_random = random.random()
            success_fail_for_q_net = False
            arr[3] += 1
            q.buffer.add(obs, 1, -1, obs_next, d)
            if q.buffer.size() > batch_size and updates[q_idx] < max_update:
                loss = q.learning(updates[0])
                losses[q_idx] = np.append(losses[q_idx], loss)
                updates[q_idx] += 1
            return idx, success_fail_for_q_net, pivot, q_net_random, next_pi, None
        else:
            arr[4] += 1
            q.buffer.add(obs, 1, 0, obs_next, d)
            return idx, False, pivot, q_net_random, cur_pi, None

def train_q_network_obstacle(args):

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
    q31 = DQN_Converter(args, dim, batch_size)
    q13 = DQN_Converter(args, dim, batch_size)
    q12.set_mean_std(pi12.get_mean(), pi12.get_std())
    q21.set_mean_std(pi21.get_mean(), pi21.get_std())
    q13.set_mean_std(pi13.get_mean(), pi13.get_std())
    q31.set_mean_std(pi31.get_mean(), pi31.get_std())

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

    q_net_pi12_random = random.random()
    q_net_pi21_random = random.random()
    q_net_pi13_random = random.random()
    q_net_pi31_random = random.random()
    success_fail_for_q_net_pi12 = False
    success_fail_for_q_net_pi21 = False
    success_fail_for_q_net_pi13 = False
    success_fail_for_q_net_pi31 = False
    temp_obs_for_q_net_pi12 = None
    temp_obs_for_q_net_pi21 = None
    temp_obs_for_q_net_pi13 = None
    temp_obs_for_q_net_pi31 = None

    const = 1.0
    intervals = []
    obstacle = env.unwrapped.get_obstacle_pos_and_type()
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
    
    idx = 0
    pivot = intervals[idx][0] + const * np.random.rand(1)
    policy = pi1


    while not (q_net_flags[0] and q_net_flags[1] and q_net_flags[2] and q_net_flags[3]):
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

        obs_next, r, d, info = env.step(a)
        
        tot_interaction += 1
        x_pos = env.unwrapped.get_x_pos()
        # env.render()
        # print(order, idx, x_pos)

        if policy == pi1 and intervals[idx][0] <= x_pos: 
            if order[idx] == 'pi12':
                policy = pi12 
            elif order[idx] == 'pi13':
                policy = pi13
        elif policy == pi12 and q_net_pi12_random <= epsilon_by_frame(updates[0]): # random action
            idx, success_fail_for_q_net_pi12, pivot, q_net_pi12_random, policy, temp_obs_for_q_net_pi12 = \
                act_random_q_net(x_pos, obs, obs_next, d, idx, intervals, pivot, q_net_pi12_random, arr1, policy, pi2, q12)
        elif policy == pi12 and q_net_pi12_random > epsilon_by_frame(updates[0]):
            idx, success_fail_for_q_net_pi12, pivot, q_net_pi12_random, policy, temp_obs_for_q_net_pi12 = \
                act_non_random_q_net(x_pos, obs, obs_next, d, idx, intervals, pivot, q_net_pi12_random, arr1, policy, pi2, q12, max_update, batch_size, updates, losses, 0)
        elif policy == pi13 and q_net_pi13_random <= epsilon_by_frame(updates[2]): # random action
            idx, success_fail_for_q_net_pi13, pivot, q_net_pi13_random, policy, temp_obs_for_q_net_pi13 = \
                act_random_q_net(x_pos, obs, obs_next, d, idx, intervals, pivot, q_net_pi13_random, arr3, policy, pi3, q13)
        elif policy == pi13 and q_net_pi13_random > epsilon_by_frame(updates[2]):
            idx, success_fail_for_q_net_pi13, pivot, q_net_pi13_random, policy, temp_obs_for_q_net_pi13 = \
                act_non_random_q_net(x_pos, obs, obs_next, d, idx, intervals, pivot, q_net_pi13_random, arr3, policy, pi3, q13, max_update, batch_size, updates, losses, 2)
        
        elif policy == pi2:
            if intervals[idx][0] <= x_pos: # 8.4
                policy = pi21 
        elif policy == pi21 and q_net_pi21_random <= epsilon_by_frame(updates[1]): # random action
            idx, success_fail_for_q_net_pi21, pivot, q_net_pi21_random, policy, temp_obs_for_q_net_pi21 = \
                act_random_q_net(x_pos, obs, obs_next, d, idx, intervals, pivot, q_net_pi21_random, arr2, policy, pi1, q21)
        elif policy == pi21 and q_net_pi21_random > epsilon_by_frame(updates[1]):
            idx, success_fail_for_q_net_pi21, pivot, q_net_pi21_random, policy, temp_obs_for_q_net_pi21 = \
                act_non_random_q_net(x_pos, obs, obs_next, d, idx, intervals, pivot, q_net_pi21_random, arr2, policy, pi1, q21, max_update, batch_size, updates, losses, 1)

        elif policy == pi3:
            if intervals[idx][0] <= x_pos: 
                policy = pi31
        elif policy == pi31 and q_net_pi31_random <= epsilon_by_frame(updates[3]): # random action
            idx, success_fail_for_q_net_pi31, pivot, q_net_pi31_random, policy, temp_obs_for_q_net_pi31 = \
                act_random_q_net(x_pos, obs, obs_next, d, idx, intervals, pivot, q_net_pi31_random, arr4, policy, pi1, q31)
        elif policy == pi31 and q_net_pi31_random > epsilon_by_frame(updates[3]):
            idx, success_fail_for_q_net_pi31, pivot, q_net_pi31_random, policy, temp_obs_for_q_net_pi31 = \
                act_non_random_q_net(x_pos, obs, obs_next, d, idx, intervals, pivot, q_net_pi31_random, arr4, policy, pi1, q31, max_update, batch_size, updates, losses, 3)



        if success_fail_for_q_net_pi12 and (info['success_count'] % 2 == 1):
            success_fail_for_q_net_pi12 = False
            arr1[5] += 1
            q12.buffer.add(temp_obs_for_q_net_pi12, 0, success_reward, obs_next, d) # successful execution
            if q12.buffer.size() > batch_size and updates[0] < max_update:
                loss = q12.learning(updates[0])
                losses[0] = np.append(losses[0], loss)
                updates[0] += 1
        elif success_fail_for_q_net_pi12 and d:
            success_fail_for_q_net_pi12 = False
            arr1[6] += 1
            q12.buffer.add(temp_obs_for_q_net_pi12, 0, fail_reward, obs_next, d) # unsuccessful execution
            if q12.buffer.size() > batch_size and updates[0] < max_update:
                loss = q12.learning(updates[0])
                losses[0] = np.append(losses[0], loss)
                updates[0] += 1
    
        if success_fail_for_q_net_pi21 and ((info['success_count'] == 1 and intervals[2][1] <= x_pos) or (info['success_count'] == 3 and intervals[6][1] <= x_pos)):
            success_fail_for_q_net_pi21 = False
            arr2[5] += 1
            q21.buffer.add(temp_obs_for_q_net_pi21, 0, success_reward, obs_next, d) # successful execution
            if q21.buffer.size() > batch_size and updates[1] < max_update:
                loss = q21.learning(updates[1])
                losses[1] = np.append(losses[1], loss)
                updates[1] += 1
        elif success_fail_for_q_net_pi21 and d:
            success_fail_for_q_net_pi21 = False
            arr2[6] += 1
            q21.buffer.add(temp_obs_for_q_net_pi21, 0, fail_reward, obs_next, d) # unsuccessful execution
            if q21.buffer.size() > batch_size and updates[1] < max_update:
                loss = q21.learning(updates[1])
                losses[1] = np.append(losses[1], loss)
                updates[1] += 1
        
        if success_fail_for_q_net_pi13 and (info['success_count'] == 2 or info['success_count'] == 4):
            success_fail_for_q_net_pi13 = False
            arr3[5] += 1
            q13.buffer.add(temp_obs_for_q_net_pi13, 0, success_reward, obs_next, d) # successful execution
            if q13.buffer.size() > batch_size and updates[2] < max_update:
                loss = q13.learning(updates[2])
                losses[2] = np.append(losses[2], loss)
                updates[2] += 1
        elif success_fail_for_q_net_pi13 and d:
            success_fail_for_q_net_pi13 = False
            arr3[6] += 1
            q13.buffer.add(temp_obs_for_q_net_pi13, 0, fail_reward, obs_next, d) # unsuccessful execution
            if q13.buffer.size() > batch_size and updates[2] < max_update:
                loss = q13.learning(updates[2])
                losses[2] = np.append(losses[2], loss)
                updates[2] += 1

        if success_fail_for_q_net_pi31 and (info['success_count'] == 3 or info['success_count'] == 5):
            success_fail_for_q_net_pi31 = False
            arr4[5] += 1
            q31.buffer.add(temp_obs_for_q_net_pi31, 0, success_reward, obs_next, d) # successful execution
            if q31.buffer.size() > batch_size and updates[3] < max_update:
                loss = q31.learning(updates[3])
                losses[3] = np.append(losses[3], loss)
                updates[3] += 1
        elif success_fail_for_q_net_pi31 and d:
            success_fail_for_q_net_pi31 = False
            arr4[6] += 1
            q31.buffer.add(temp_obs_for_q_net_pi31, 0, fail_reward, obs_next, d) # unsuccessful execution
            if q31.buffer.size() > batch_size and updates[3] < max_update:
                loss = q31.learning(updates[3])
                losses[3] = np.append(losses[3], loss)
                updates[3] += 1



        obs = obs_next

        if d:
            obs = env.reset()
            intervals = []
            obstacle = env.unwrapped.get_obstacle_pos_and_type()
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

            idx = 0
            pivot = intervals[idx][0] + const * np.random.rand(1)
            policy = pi1

        if updates[0] >= max_update:
            q_net_flags[0] = True
        if updates[1] >= max_update:
            q_net_flags[1] = True
        if updates[2] >= max_update:
            q_net_flags[2] = True
        if updates[3] >= max_update:
            q_net_flags[3] = True

        save_qnet(prev_updates, updates, fpath, q12, q21, q13, q31, save_frequency)

        for i in range(len(updates)):
            prev_updates[i] = updates[i]

        if(sum(updates) % 2000 == 0):
            print(updates)
            print(arr1)
            print(arr2)
            print(arr3)
            print(arr4)
        
    print('obstacle tot_interaction:', tot_interaction)
    f.close()
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    args = p.parse_args()
    train_q_network_obstacle(args)

