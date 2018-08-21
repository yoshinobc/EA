import gym
import deap
import random
from deap import base,creator
from deap import tools

env=gym.make("CartPole-v0")

for episode in range(10):
    env.reset()
    for t in range(100):
        action=env.action_space.sample()
        observation,reward,done,info=env.step(action)
        env.render()
        print("\nobservation=",observation)
        print("\nreward=",reward)
        print("\ndone=",done)
        print("\ninfo=",info)
        if(abs(observation[2]) >0.78):
            break
        env.monitor_close()
