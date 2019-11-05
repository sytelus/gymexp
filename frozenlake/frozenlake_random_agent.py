import time
import random
import numpy as np
import gym
import frozenlake.utils as fzutils

fzutils.seed(42)
env = fzutils.get_env(40)

def run_episode(env):
    obs = env.reset()
    reward, env_done, steps, total_r = 0.0, False, 0, 0.0
    while not env_done:
        action = env.action_space.sample()
        obs, reward, env_done, info = env.step(action=action)
        total_r += reward
        steps += 1
    return reward, steps

start_time = time.time()
print('Started', start_time)

episode_count, max_reward, step_count = 0, -1.0E16, 0
while max_reward < 1.0:
    reward, steps = run_episode(env)
    episode_count += 1
    max_reward = max(max_reward, reward)
    step_count += steps
    if episode_count % 100000 == 0:
        print('Progress: episode_count, max_reward, step_count, time', 
            episode_count, max_reward, step_count, time.time() - start_time)        

print('Done: episode_count, max_reward, step_count, time', 
    episode_count, max_reward, step_count, time.time() - start_time)




