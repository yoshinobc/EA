
# coding: utf-8

# In[1]:


import deap
#import NN
from NN import NNN
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

nn = NNN()
ENV = gym.make("CartPole-v0")
MAX_STEPS = 200
NGEN = 30
w_list = []
# In[9]:

creator.create("FitnessMax",base.Fitness,weights=[1.0])
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

def decide_weight(network):
    #print(np.reshape(network['W1'],(12)))
    w_list = []
    w_list.extend(np.reshape(network['W1'],(12)))
    w_list.extend(np.reshape(network['B1'],(3)))
    w_list.extend(np.reshape(network['W2'],(6)))
    w_list.extend(np.reshape(network['B2'],(2)))
    w_list.extend(np.reshape(network['W3'],(2)))
    w_list.extend(np.reshape(network['B3'],(1)))
    '''
    w_list.extend(np.reshape(NNN.network['W1'],(12)))
    w_list.extend(np.reshape(NNN.network['B1'],(3)))
    w_list.extend(np.reshape(NNN.network['W2'],(6)))
    w_list.extend(np.reshape(NNN.network['B2'],(2)))
    w_list.extend(np.reshape(NNN.network['W3'],(2)))
    w_list.extend(np.reshape(NNN.network['B3'],(1)))
    '''
    return w_list

toolbox.register("gene",decide_weight,nn.network)
toolbox.register("individual",tools.initIterate,creator.Individual,toolbox.gene)
toolbox.register("population",tools.initRepeat,list,toolbox.individual)

def EV(individual):
    observation = ENV.reset()
    episode_reward = 0
    if not individual :
        print("kara")
        for i in range(26):
            individual.append(random.randrange(10))
        return episode_reward

    nn.update(individual);

    for step in range(MAX_STEPS):
        action = get_action(observation,individual)
        observation_next,reward,done,info_not = ENV.step(action)

        if done:
            ENV.render()
            break

        observation = observation_next
        episode_reward += reward

        ENV.render()

    return [episode_reward]

toolbox.register("evaluate",EV)
toolbox.register("mate",tools.cxTwoPoint)　#float
toolbox.register("mutate",tools.mutFlipBit,indpb=0.05) #mutFllipBit floatに対して津えるやつ
toolbox.register("select",tools.selTournament,tournsize=3)

def get_action(observation,individual):
    action = decide_action(observation,individual)
    return action

def decide_action(observation,individual):
    return nn.conclusion(observation,individual)

def main(nn):
    random.seed(1)
    #nn.init_network(NNN);
    #nn = NN()

    pop = toolbox.population(n=100)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind:ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    pop,log = algorithms.eaSimple(pop,toolbox,cxpb=0.5,mutpb=0.2,ngen=40,stats=stats,halloffame=hof,verbose=True)

    return pop,log,hof
if __name__=='__main__':
    pop,log,hof = main(nn)
