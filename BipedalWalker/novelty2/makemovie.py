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
MAX_STEPS = 1000
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
    observation = ENV.reset()
    action_lists = []
    total_novelty = 0
    for i in range(MAX_STEPS):
        #ENV = wrappers.Monitor(env,'/mnt/c/Users/bc/Documents/EA/BipedalWalker/noveltymovies/',force=True)
        action = get_action(observation,network)
        observation,reward,done,xylists = ENV.step(action)
        ENV.render()
        action_lists.append(xylists)
        total_reward += reward
        if done:
            ENV.render()
            break
    for i in range(len(action_lists) - 1):
            total_novelty += np.sqrt((action_lists[i+1][0] - action_lists[i][0])**2 + (action_lists[i+1][1] - action_lists[i][1])**2)
    print(count,":",total_novelty)
    count+=1
def main():
    '''
    with open('gen150checkpoint_maxstepsliner', 'rb') as f:
        population = pickle.load(f)
    '''
    with open('gen1000hof','rb') as f:
        ind = pickle.load(f)
    print(fp)
    for ind in population:
        rendering(ind)
if __name__=='__main__':
    main()
