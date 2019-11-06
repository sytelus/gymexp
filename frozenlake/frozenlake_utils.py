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
    
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if reward == 0.0:
            # lower values for done will cause suicide tendency for agent
            # without else part, agent might just oscillate forever
            if done:
                reward = -1.0E4  
            else:
                col = obs % self.ncols
                row = (obs - col) // self.nrows
                dist = abs(self.ncols - col - 1) + abs(self.nrows -row -1)
                dist_norm = float(dist) / (self.ncols + self.nrows) * 0.1
                #print(obs, col, row, dist, dist_norm, self.ncols)
                reward =  -dist_norm

                
        return obs, reward, done, info