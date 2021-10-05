import argparse
import os


def argparser():
    def str2bool(v):
        return v.lower() == 'true'

    def str2list(v):
        if not v:
            print(v)
            return v
        else:
            return [v_ for v_ in v.split(',')]

    parser = argparse.ArgumentParser(description='Transition Policy with Inverse Reinforcement Learning',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--mpi', type=int, default=1, help="use mpi")

    parser.add_argument('--train', type=str2bool, default=False)
    parser.add_argument('--hrl', type=str2bool, default=False)
    parser.add_argument('--complex_task', type=str, default='patrol', choices=['patrol', 'obstacle', 'hurdle', 'pick', 'serve', 'catch'])
    
    parser.add_argument('--seed', type=int, default=5000)
    parser.add_argument('--env', type=str, default='Walker2dForward-v1')
    parser.add_argument('--exp_name', type=str, default='Hopper-v3')
    
    parser.add_argument('--epochs', type=int, default=4000)
    parser.add_argument('--num_rollouts', type=int, default=8000)
    parser.add_argument('--max_steps', type=int, default=8000)
    


    parser.add_argument('--primitive_algo', type=str, default='ppo',
                        choices=['ppo', 'trpo', 'ppo_reward_test'])
    parser.add_argument('--primitive_hid_size', type=int, default=64)
    parser.add_argument('--primitive_hid_layer', type=int, default=2)
    parser.add_argument('--save_freq', type=int, default=10)

    # Lee trpo
    parser.add_argument('--rl_num_hid_layers', type=int, default=2)
    parser.add_argument('--rl_hid_size', type=int, default=32)
    parser.add_argument('--rl_fixed_var', type=str2bool, default=True)
    parser.add_argument('--rl_activation', type=str, default='tanh', choices=['relu', 'elu', 'tanh'])
    parser.add_argument('--primitive_include_acc', type=str2bool, default=False)

    # ppo
    parser.add_argument('--lr_decay', type=str2bool, default=True)
    parser.add_argument('--clip_ratio', type=float, default=0.20)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lam', type=float, default=0.97)
    parser.add_argument('--target_kl', type=float, default=0.015)
    parser.add_argument('--entcoeff', type=float, default=0.002)
    parser.add_argument('--ppo_pi_lr', type=float, default=1e-4)
    parser.add_argument('--ppo_vf_lr', type=float, default=1e-4)
    parser.add_argument('--ppo_batchsize', type=int, default=64)
    parser.add_argument('--ppo_train_v_iters', type=int, default=10)
    
    # trpo
    parser.add_argument('--delta', type=float, default=0.01)
    parser.add_argument('--damping_coeff', type=float, default=0.1)
    parser.add_argument('--cg_iters', type=int, default=10)
    parser.add_argument('--backtrack_coeff', type=float, default=1.0)
    parser.add_argument('--backtrack_iters', type=int, default=10)
    parser.add_argument('--trpo_batchsize', type=int, default=512)
    parser.add_argument('--trpo_train_v_iters', type=int, default=10)

    # sac
    parser.add_argument('--sac_steps_per_epoch', type=int, default=8000)
    parser.add_argument('--sac_epochs', type=int, default=8000)
    parser.add_argument('--sac_replay_size', type=int, default=int(1e6))
    parser.add_argument('--sac_gamma', type=float, default=0.99)

    parser.add_argument('--sac_polyak', type=float, default=0.995)
    parser.add_argument('--sac_lr', type=float, default=1e-3)
    parser.add_argument('--sac_alpha', type=float, default=0.2)
    parser.add_argument('--sac_batchsize', type=int, default=100)
    parser.add_argument('--sac_start_steps', type=int, default=10000)

    parser.add_argument('--sac_update_after', type=int, default=1000)
    parser.add_argument('--sac_update_every', type=int, default=50)
    parser.add_argument('--sac_num_test_episodes', type=int, default=10)
    parser.add_argument('--sac_max_ep_len', type=int, default=8000)

    parser.add_argument('--sac_save_freq', type=int, default=1)

    # collect expert demonstration 
    parser.add_argument('--collect_exp_data', type=str2bool, default=False)
    parser.add_argument('--front', type=str2bool, default=True)
    # parser.add_argument('--exp_buffer_size', type=int, default=5000)
    parser.add_argument('--exp_buffer_size', type=int, default=1000000)
    parser.add_argument('--suffix', type=str, default='none')
    parser.add_argument('--primitive_path', type=str, default='data/ppo_forward/ppo_forward_s5003/pyt_save/model.pt')
    parser.add_argument('--primitive_env', type=str)
    parser.add_argument('--primitive', type=str, choices=['walk', 'jump', 'crawl'])

    # irl
    parser.add_argument('--irl_training', type=str2bool, default=False)
    parser.add_argument('--env_1', type=str, default='forward')
    parser.add_argument('--env_2', type=str, default='jump')
    parser.add_argument('--exp_data_path_1', type=str, default='data/exp_demo/Walker2dForwardHmap-v1')
    parser.add_argument('--exp_data_path_2', type=str, default='data/exp_demo/Walker2dJumpHmap-v1')
    parser.add_argument('--irl_algo', type=str, default='airl')

    # evaluate path
    parser.add_argument('--path', type=str, default='none')
    parser.add_argument('--render', type=str2bool, default=False)
    parser.add_argument('--eval_algo', type=str, default='trpo', choices=['ppo', 'trpo'])

    # Q-network
    parser.add_argument('--gamma_q', type=float, default=0.99)
    parser.add_argument('--learning_rate_q', type=float, default=1e-4)
    parser.add_argument('--max_buffer_size_q', type=int, default=1000000)
    parser.add_argument('--update_target_frequency_q', type=int, default=500)
    parser.add_argument('--epsilon_q', type=int, default=1)
    parser.add_argument('--epsilon_min_q', type=float, default=0.01)
    parser.add_argument('--epsilon_decay_q', type=int, default=500)
    parser.add_argument('--epoch_q', type=int, default=20000)
    parser.add_argument('--batch_size_q', type=int, default=64)
    parser.add_argument('--eval_frequency_q', type=int, default=1000)

    # training Q-network for obstacle and patorl single
    parser.add_argument('--option', type=str, default='p', choices=['n', 'p']) # negative direction first 
    parser.add_argument('--prev', type=str, default='none')
    parser.add_argument('--tran', type=str, default='none')
    parser.add_argument('--next', type=str, default='none')
    parser.add_argument('--fname', type=str, default='jump')

    # training Q-network for obstacle and patorl
    parser.add_argument('--pi1', type=str, default='none') # obstacle: walk / patrol: forward walk
    parser.add_argument('--pi2', type=str, default='none') # obstacle: jump / patrol: backward walk
    parser.add_argument('--pi3', type=str, default='none') # obstacle: crawl
    parser.add_argument('--pi12', type=str, default='none') # obstacle: walk->jump / patrol: forward walk-> backward walk
    parser.add_argument('--pi21', type=str, default='none') # obstacle: jump->walk / patrol: backward walk-> forward walk
    parser.add_argument('--pi13', type=str, default='none') # obstacle: walk->crawl
    parser.add_argument('--pi31', type=str, default='none') # obstacle: crawl->walk
    parser.add_argument('--pi23', type=str, default='none') 
    parser.add_argument('--pi32', type=str, default='none') 
    parser.add_argument('--q12', type=str, default='none') # obstacle Q-network: walk->jump / patrol Q-network: forward walk-> backward walk
    parser.add_argument('--q21', type=str, default='none') # obstacle Q-network: jump->walk / patrol Q-network: backward walk-> forward walk
    parser.add_argument('--q13', type=str, default='none') # obstacle Q-network: walk->crawl
    parser.add_argument('--q31', type=str, default='none') # obstacle Q-network: crawl->walk
    parser.add_argument('--q23', type=str, default='none') 
    parser.add_argument('--q32', type=str, default='none') 
    parser.add_argument('--single', type=str2bool, default=False)
    parser.add_argument('--pi1_env', type=str, default='none') # obstacle: walk / patrol: forward walk
    parser.add_argument('--pi2_env', type=str, default='none') # obstacle: jump / patrol: backward walk
    parser.add_argument('--pi3_env', type=str, default='none') # obstacle: jump / patrol: backward walk

    parser.add_argument('--perplexity', type=int, default=40)
    args = parser.parse_args()

    return args
