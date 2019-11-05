import gym
import ray
import numpy as np
from ray.rllib.agents.dqn import DQNTrainer, DEFAULT_CONFIG
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env
import frozenlake.utils as fzutils


fzutils.seed(42)

ray.init(num_gpus=1)

config = DEFAULT_CONFIG.copy()
# config.update({
#     "gamma": 0.99,
#     "lr": 0.0001,
#     "learning_starts": 10000,
#     "buffer_size": 50000,
#     "sample_batch_size": 4,
#     "train_batch_size": 320,
#     "schedule_max_timesteps": 2000000,
#     "exploration_final_eps": 0.01,
#     "exploration_fraction": 0.1,

#     "model": {"dim":64}
#     })

def env_creator(env_config):
    env = fzutils.get_env(40, one_hot_obs=True)
    return env

register_env("frozenworld_env", env_creator)
agent = DQNTrainer(config=config, env="podworld_env")

for i in range(500):
    stats = agent.train()
    # print(pretty_print(stats))
    print ('episode_reward_mean', stats['episode_reward_min'])
