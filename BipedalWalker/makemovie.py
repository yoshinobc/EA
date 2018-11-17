import deap
#import NN
from walker_NN import NNN
from deap import base
from deap import creator
from deap import tools
import math
import gym
import random
from gym import wrappers
import time
import numpy as np
from deap import algorithms
import operator
import pickle

ENV = gym.make("BipedalWalker-v2")
MAX_STEPS = 100
count = 0
creator.create("FitnessMax",base.Fitness,weights=[1.0])
creator.create("Individual", list, fitness=creator.FitnessMax)
nn = NNN()
def get_action(observation,network):
    action = decide_action(observation,network)
    return action

def decide_action(observation,network):
    return nn.conclusion(observation,network)

def rendering(individual):
    global env
    global count
    network = nn.update(individual); #networkにindividualを適用
    total_reward = 0
    for i in range(MAX_STEPS):
        #ENV = wrappers.Monitor(env,'/mnt/c/Users/bc/Documents/EA/BipedalWalker/noveltymovies/',force=True)
        observation = ENV.reset()
        action = get_action(observation,network)
        observation,reward,done,xylists = ENV.step(action)
        print("action:",action,"xylists:",xylists)
        ENV.render()
        total_reward += reward
        if done:
            ENV.render()
            break
    print(count,":",total_reward)
    count+=1
def main():
    with open('gen500checkpoints', 'rb') as f:
        population = pickle.load(f)
    for ind in population:
        rendering(ind)
if __name__=='__main__':
    main()
