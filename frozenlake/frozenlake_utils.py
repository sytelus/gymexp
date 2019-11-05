import random
import numpy as np
import gym
from gym.envs.toy_text.frozen_lake import generate_random_map

def seed(s):
    np.random.seed(s)
    random.seed(s)

def get_env(size, p=0.88, one_hot_obs=True, neg_dead_rew=True):
    random_map = generate_random_map(size=size, p=p)
    env = gym.make("FrozenLake-v0", desc=random_map)
    if one_hot_obs:
        env = Int2OneHotWrapper(env)
    if neg_dead_rew:
        env = NegativeOnDeadWrapper(env)
    return env 


class Int2OneHotWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super(Int2OneHotWrapper, self).__init__(env)
        assert isinstance(env.observation_space, gym.spaces.Discrete)
        # convert integer to one hot array
        self.observation_space = gym.spaces.Box(0.0, 1.0, (env.observation_space.n, ), dtype=np.float32)

    def observation(self, observation):
        new_obs = np.copy(self.observation_space.low)
        new_obs[observation] = 1.0
        return new_obs

class NegativeOnDeadWrapper(gym.Wrapper):
    def __init__(self, env):
        super(NegativeOnDeadWrapper, self).__init__(env)
    
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if done and reward == 0.0:
            reward = -1.0
        return obs, reward, done, info