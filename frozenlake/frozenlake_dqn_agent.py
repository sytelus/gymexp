import gym
import ray
import numpy as np
#from ray.rllib.agents.dqn import DQNTrainer, DEFAULT_CONFIG
import ray.rllib.agents.ppo as ppo
import ray.rllib.agents.dqn as dqn
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env
import frozenlake_utils as fzutils
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--seed', action="store", type=int, default=42)
parser.add_argument('--density', action="store", type=int, default=1)
parser.add_argument('--size', action="store", type=int, default=40)
parser.add_argument('--slippery', action="store", type=int, default=1)
parser.add_argument('--algo', action="store", type=str, default='dqn')
args = parser.parse_args()

fzutils.seed(args.seed)

ray.init(num_gpus=1)

#config = DEFAULT_CONFIG.copy()
config = {
    # "timesteps_per_iteration": 100,
    # "target_network_update_freq": 50,
    # "buffer_size": 5000,
    # "lr": 5e-3,
    # "learning_starts": 100,

    # "noisy": True,
    # "exploration_fraction": 0.1,
    # "exploration_final_eps": 0.01,
    # "schedule_max_timesteps": 200000,

    "compress_observations": False,
    "num_workers": 1,
    "num_gpus": 1    
    }

def env_creator(env_config):
    env = fzutils.get_env(size=args.size, 
        p=(1.0 - float(args.density)/args.size),
        is_slippery=args.slippery > 0)
    return env

register_env("frozenworld_env", env_creator)
algos = {'ppo': ppo.PPOTrainer, 'dqn': dqn.DQNTrainer}
algo =  algos[args.algo] 
agent = algo(config=config, env="frozenworld_env")

for i in range(50000):
    stats = agent.train()
    s = pretty_print(stats)
    print(s)
    print ('i, episode_reward_mean, episode_len_mean', i, stats['episode_reward_mean'], stats['episode_len_mean'])
    if stats['episode_reward_min'] > 0.0:
        s = pretty_print(stats)
        print(s, file=open('./result_{}_{}_{}.txt'.format(args.slippery, args.algo, args.density), 'w'))
        exit(0)
