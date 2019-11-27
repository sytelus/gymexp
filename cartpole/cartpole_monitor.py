import gym

env = gym.make('CartPole-v0')
env = gym.wrappers.Monitor(env, 'vid', force=True)
obs = env.reset()
r, done, i = 0.0, False, 0
while not done:
    action = env.action_space.sample()
    obs, reward, done, _ = env.step(action)
    r += reward
    i += 1

print('Reward, iter', r, i)
