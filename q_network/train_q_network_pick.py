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


def save_qnet(prev_updates, updates, fpath, q12, save_frequency):
    
    if prev_updates[0] != updates[0]:
        if updates[0] % save_frequency == 0:
            fname = str(updates[0]) + '_q12.pt'
            q12.save_model(fpath, fname)

def act_random_q_net(cur_duration, obs, obs_next, d, pivot, q_random, arr, cur_pi, next_pi, q, duration):
    if cur_duration < pivot: # stay
        arr[1] += 1
        q.buffer.add(obs, 1, 0, obs_next, d)
        return False, pivot, q_random, cur_pi, obs_next
    elif pivot <= cur_duration:
        pivot = np.random.randint(duration) + 1
        q_net_pi12_random = random.random()
        temp_obs_for_q_net_pi12 = obs_next # keep observation
        return True, pivot, q_net_pi12_random, next_pi, temp_obs_for_q_net_pi12

def act_non_random_q_net(cur_duration, obs, obs_next, d, pivot, q_net_random, arr, cur_pi, next_pi, q, max_update, batch_size, updates, losses, q_idx, info, duration):
    q_a = q.act(obs_next)
    if q_a == 0: # convertd
        pivot = np.random.randint(duration) + 1
        q_net_random = random.random()
        success_fail_for_q_net = True
        temp_obs_for_q_net_pi12 = obs_next # keep observation
        return success_fail_for_q_net, pivot, q_net_random, next_pi, temp_obs_for_q_net_pi12
    elif q_a == 1: # stay
        if d and info["success_count"] != 5:
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

def train_q_network_pick(args):

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

    dim = 1
    for d in env.observation_space.shape:
        dim = dim * d

    batch_size = 64 #args.batch_size_q
    eval_frequency = args.eval_frequency_q

    q12 = DQN_Converter(args, dim, batch_size)
    q12.set_mean_std(pi12.get_mean(), pi12.get_std())

    # non-Linear epsilon decay
    epsilon_final = args.epsilon_min_q
    epsilon_start = args.epsilon_q
    epsilon_decay = args.epsilon_decay_q
    epsilon_by_frame = lambda epoch_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(
        -1. * epoch_idx / epsilon_decay)

    fail_reward = -1.0
    success_reward = 1.0
    losses = [np.array([])]
    updates = [0]
    prev_updates = [-1]
    max_update = 15000

    tot_interaction = 0
    arr1 = [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0] 
    q_net_flags = [False]
    
    eval_frequency = 250
    save_frequency = 1000

    fpath = osp.join('data', 'q_network', args.fname)
    os.makedirs(fpath, exist_ok=True)
    fname = osp.join(fpath, 'progress.txt')
    f = open(fname, 'w')
    f.write("total_interaction num_success num_update1 num_update2 force_1_to_2 force_2_to_1 loss1_mean, loss2_mean\n")

    policy = pi1
    obs = env.reset()

    q_net_pi12_random = random.random()
    success_fail_for_q_net_pi12 = False
    temp_obs_for_q_net_pi12 = None
    duration = 30
    pivot = np.random.randint(duration) + 1

    cur_success_count = 1
    cur_duration = 0
    
    while not (q_net_flags[0]):
        if policy == pi1:
            # print('pi1')
            a, vpred = policy.act(obs, False)
            cur_duration = 0
        elif policy == pi12:
            # print('pi12')
            a = policy.exploit(obs)
            cur_duration += 1
        obs_next, r, d, info = env.step(a)
        # env.render()
        # time.sleep(1e-2)
        tot_interaction += 1
        # print('x_pos: ', x_pos, 'pivot: ', pivot, 'curb_pos: ', env.unwrapped.get_curb_pos())

#################################################################################################################
        
        if policy == pi1 and env.unwrapped.is_terminate(): 
            policy = pi12 
            print('policy pi12')
        elif policy == pi12 and q_net_pi12_random <= epsilon_by_frame(updates[0]): # random action
            # print('pi12 random action')
            success_fail_for_q_net_pi12, pivot, q_net_pi12_random, policy, temp_obs_for_q_net_pi12 = \
                act_random_q_net(cur_duration, obs, obs_next, d, pivot, q_net_pi12_random, arr1, policy, pi1, q12, duration)
        elif policy == pi12 and q_net_pi12_random > epsilon_by_frame(updates[0]):
            # print('pi12 non random action')
            success_fail_for_q_net_pi12, pivot, q_net_pi12_random, policy, temp_obs_for_q_net_pi12 = \
                act_non_random_q_net(cur_duration, obs, obs_next, d, pivot, q_net_pi12_random, arr1, policy, pi1, q12, max_update, batch_size, updates, losses, 0, info, duration)
        if success_fail_for_q_net_pi12 and cur_success_count < info['success_count']:
            cur_success_count = info['success_count']
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
        
#################################################################################################################

        obs = obs_next

        if d:
            obs = env.reset()
            policy = pi1
            cur_success_count = 1
            cur_duration = 0
            pivot = np.random.randint(duration) + 1


        if updates[0] >= max_update:
            q_net_flags[0] = True
        print(arr1)
        save_qnet(prev_updates, updates, fpath, q12, save_frequency)

        for i in range(len(updates)):
            prev_updates[i] = updates[i]

    print('pick tot_interaction:', tot_interaction)
    f.close()

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    args = p.parse_args()
    train_q_network_pick(args)

