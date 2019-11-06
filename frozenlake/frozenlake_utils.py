import random
import numpy as np
import gym
import math
from gym.envs.toy_text.frozen_lake import generate_random_map

def seed(s):
    np.random.seed(s)
    random.seed(s)

def get_env(size, p=0.88, one_hot_obs=True, neg_dead_rew=True):
    random_map = generate_random_map(size=size, p=p)
    env = gym.make("FrozenLake-v0", desc=random_map)
    if neg_dead_rew:
        env = NegativeOnDeadWrapper(env)    
    if one_hot_obs:
        env = Int2OneHotWrapper(env)
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
        self.ncols = self.nrows = int(math.sqrt(env.observation_space.n))

    def reset(self, **kwargs):
        self.last_obs = self.env.reset(**kwargs) 
        self.last_dist = self._get_dist(self.last_obs)
        return self.last_obs   

    def _get_dist(self, obs):
        col = obs % self.ncols
        row = (obs - col) // self.nrows
        dist = abs(self.ncols - col - 1) + abs(self.nrows -row -1)
        return dist

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        dist = self._get_dist(obs)
        if reward == 0.0:
            # lower values for done will cause suicide tendency for agent
            # without else part, agent might just oscillate forever
            if done:
                reward = -1.0E1  
            else:
                reward =  self.last_dist - dist
        else:
            reward = 1.0E1

        self.last_dist = dist
        self.last_obs = obs  

        return obs, reward, done, info