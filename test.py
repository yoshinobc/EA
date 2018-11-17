from __future__ import print_function
import os
import neat
import gym
import numpy as np
from gym import wrappers
import pickle

def ReLU(x):
    y = np.maximum(0, x)
    return y

def get_action(observation):
    action = ReLU(0.5066 + observation[3] * 0.5431769528016781 + observation[2] * 1.621185868659596 + observation[1] * 0.6335653727818767 + observation[0] * 0.323395443300985)
    if action > 0.5:
        action = 1
    else :
        action = 0
    return action

ENV = gym.make("CartPole-v1")
observation = ENV.reset()
episode_reward = 0
for step in range(500):
    action = get_action(observation)
    observation_next,reward,done,info_not = ENV.step(action)
    episode_reward += reward
    if done:
        print(episode_reward)
        ENV.render()
        break

    observation = observation_next
    #if Myclass.count>=2700:
    #    env = wrappers.Monitor(env,'./movie/cartpole-experiment-1')
    ENV.render()
