import gym
import podworld

env = gym.make('podworld-v0') #BreakoutDeterministic-v4, BreakoutNoFrameskip-v4, PongNoFrameskip-v4
env.reset()
 
import time
st = time.time()

for _ in range(10000):
    #env.render()
    env.step(env.action_space.sample())

print('headless fps', 10000.0/(time.time()-st))

env.close()