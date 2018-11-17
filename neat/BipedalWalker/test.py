import gym

env = gym.make('BipedalWalker-v2')
episods = 10
steps = 1600
for episode in range(episods):
    observation = env.reset()
    for step in range(steps):
        env.render()
        action = env.action_space.sample()
        print(action)
        observation,reward,done,info = env.step(action)
        if done:
            break
