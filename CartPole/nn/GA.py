
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
ENV = gym.make("CartPole-v1")
MAX_STEPS = 500
w_list = []
# In[9]:

creator.create("FitnessMax",base.Fitness,weights=[1.0])
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

def decide_weight(network):
    #print(np.reshape(network['W1'],(12)))
    w_list = []
    if not network :
        """
        w_list.extend(np.reshape(np.random.normal(0,1,(4,3)),(12)))
        w_list.extend(np.reshape(np.random.normal(0,1,(1,3)),(3)))
        w_list.extend(np.reshape(np.random.normal(0,1,(3,2)),(6)))
        w_list.extend(np.reshape(np.random.normal(0,1,(1,2)),(2)))
        w_list.extend(np.reshape(np.random.normal(0,1,(2,1)),(2)))
        w_list.extend(np.reshape(np.random.normal(0,1,(1,1)),(1)))
        """
        w_list.extend(np.reshape(np.random.normal(0,1,(24,16)),(384)))
        w_list.extend(np.reshape(np.random.normal(0,1,(1,16)),(16)))
        w_list.extend(np.reshape(np.random.normal(0,1,(16,8)),(128)))
        w_list.extend(np.reshape(np.random.normal(0,1,(1,8)),(8)))
        w_list.extend(np.reshape(np.random.normal(0,1,(8,4)),(32)))
        w_list.extend(np.reshape(np.random.normal(0,1,(1,4)),(4)))
        """
        w_list.extend(np.reshape(np.random.normal(0,1,(4,6)),(24)))
        w_list.extend(np.reshape(np.random.normal(0,1,(1,6)),(6)))
        w_list.extend(np.reshape(np.random.normal(0,1,(6,5)),(30)))
        w_list.extend(np.reshape(np.random.normal(0,1,(1,5)),(5)))
        w_list.extend(np.reshape(np.random.normal(0,1,(5,3)),(15)))
        w_list.extend(np.reshape(np.random.normal(0,1,(1,3)),(3)))
        w_list.extend(np.reshape(np.random.normal(0,1,(3,1)),(3)))
        w_list.extend(np.reshape(np.random.normal(0,1,(1,1)),(1)))
        """
    """
    else :
        w_list.extend(np.reshape(network['W1'],(12)))
        w_list.extend(np.reshape(network['B1'],(3)))
        w_list.extend(np.reshape(network['W2'],(6)))
        w_list.extend(np.reshape(network['B2'],(2)))
        w_list.extend(np.reshape(network['W3'],(2)))
        w_list.extend(np.reshape(network['B3'],(1)))
    """
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

    network = nn.update(individual); #networkにindividualを適用

    for step in range(MAX_STEPS):
        action = get_action(observation,individual,network)
        observation_next,reward,done,info_not = ENV.step(action)

        if done:
            ENV.render()
            break

        observation = observation_next
        episode_reward += reward

        ENV.render()

    return [episode_reward]

toolbox.register("evaluate",EV)
toolbox.register("mate",tools.cxBlend,alpha=0.5) #float
toolbox.register("mutate",tools.mutGaussian,mu=0,sigma=0.5,indpb=0.05) #mutFllipBit floatに対して津えるやつ
toolbox.register("select",tools.selTournament,tournsize=3)

def get_action(observation,individual,network):
    action = decide_action(observation,individual,network)
    return action

def decide_action(observation,individual,network):
    return nn.conclusion(observation,individual,network)

def main():
    random.seed(1)

    pop = toolbox.population(n=400)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind:ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    pop,log = algorithms.eaSimple(pop,toolbox,cxpb=0.5,mutpb=0.2,ngen=100,stats=stats,halloffame=hof,verbose=True)

    return pop,log,hof
if __name__=='__main__':
    print("pop_num = ",400)
    print("gen_num ",100)
    pop,log,hof = main()
    best_ind = tools.selBest(pop,1)[0]
    for ind in hof :
        print(ind ,end="")
        print(ind.fitness)
    print(nn.update(best_ind))
