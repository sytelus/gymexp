import gym
import ray
from ray.rllib.agents.ppo import PPOTrainer, DEFAULT_CONFIG
from ray.tune.logger import pretty_print

import tensorflow
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

from torch import cuda
print(cuda.is_available())
print(cuda.device_count())

ray.init(num_gpus=2)
print(ray.get_gpu_ids())

config = DEFAULT_CONFIG.copy()
config['num_workers'] = 1
config['num_sgd_iter'] = 30
config['sgd_minibatch_size'] = 128
config['model']['fcnet_hiddens'] = [100, 100]
config['num_cpus_per_worker'] = 0  # This avoids running out of resources in the notebook environment when this cell is re-executed

agent = PPOTrainer(config, 'CartPole-v0')
