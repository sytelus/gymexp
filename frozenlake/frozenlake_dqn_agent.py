import gym
import ray
import numpy as np
from ray.rllib.agents.dqn import DQNTrainer, DEFAULT_CONFIG
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env
import frozenlake_utils as fzutils


fzutils.seed(42)

ray.init(num_gpus=1)

config = DEFAULT_CONFIG.copy()
config.update({
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
    })

def env_creator(env_config):
    env = fzutils.get_env(40)
    return env

register_env("frozenworld_env", env_creator)
agent = DQNTrainer(config=config, env="frozenworld_env")

for i in range(50000):
    stats = agent.train()
    print(pretty_print(stats))
    print ('i, episode_reward_mean, episode_len_mean', i, stats['episode_reward_mean'], stats['episode_len_mean'])
    if stats['episode_reward_min'] > 0.0:
        input("arrived?")
