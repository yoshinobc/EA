
# coding: utf-8

# In[1]:


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
import pickle
NGEN=300
env = gym.make("BipedalWalker-v2")
nn = NNN()
MAX_STEPS = 3700
w_list = []
# In[9]:
count = 0
CXPB=0.5
MUTPB=0.1
creator.create("FitnessMax",base.Fitness,weights=[1.0],)
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

def decide_weight(network):
    #print(np.reshape(network['W1'],(12)))
    w_list = []
    if not network :
        """
        w_list.extend(np.reshape(np.random.normal(0,1,(24,16)),(384)))
        w_list.extend(np.reshape(np.random.normal(0,1,(1,16)),(16)))
        w_list.extend(np.reshape(np.random.normal(0,1,(16,8)),(128)))
        w_list.extend(np.reshape(np.random.normal(0,1,(1,8)),(8)))
        w_list.extend(np.reshape(np.random.normal(0,1,(8,4)),(32)))
        w_list.extend(np.reshape(np.random.normal(0,1,(1,4)),(4)))
        """
        w_list.extend(np.reshape(np.random.normal(0,1,(24,10)),(240)))
        w_list.extend(np.reshape(np.random.normal(0,1,(1,10)),(10)))
        w_list.extend(np.reshape(np.random.normal(0,1,(10,4)),(40)))
        w_list.extend(np.reshape(np.random.normal(0,1,(1,4)),(4)))
    return w_list

toolbox.register("gene",decide_weight,nn.network)
toolbox.register("individual",tools.initIterate,creator.Individual,toolbox.gene)
toolbox.register("population",tools.initRepeat,list,toolbox.individual)

def EV(individual):
    global count
    network = nn.update(individual); #networkにindividualを適用
    final_reward = 0
    for _ in range(3):
        episode_reward = 0
        observation = env.reset()
        while True:
            action = get_action(observation,network)
            observation,reward,done,info = env.step(action)
            episode_reward += reward
            #if (-4.8  > observation[0]) or (observation[0] > 4.8) or (0.017453292519943 < observation[3] < -0.017453292519943) or (episode_reward >= MAX_STEPS):
            if done:
                break
        final_reward += episode_reward
    return final_reward / 3,

toolbox.register("evaluate",EV)
toolbox.register("mate",tools.cxBlend,alpha=0.5) #float
toolbox.register("mutate",tools.mutGaussian,mu=0,sigma=0.5,indpb=0.05) #mutFllipBit floatに対して津えるやつ
toolbox.register("select",tools.selTournament,tournsize=3)

def get_action(observation,network):
    action = decide_action(observation,network)
    return action

def decide_action(observation,network):
    return nn.conclusion(observation,network)

def main():
    random.seed(1)

    pop = toolbox.population(n=250)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind:ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    #pop,log, = algorithms.eaSimple(pop,toolbox,cxpb=0.5,mutpb=0.2,ngen=200,stats=stats,halloffame=hof,verbose=True)

    fitness = list(map(toolbox.evaluate,pop))

    for ind, fit in zip(pop, fitness):
        ind.fitness.values = fit

    print("gen","max","mean")
    for i in range(NGEN+1):
        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))

        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):

            # cross two individuals with probability CXPB
            if random.random() < CXPB:
                toolbox.mate(child1, child2)

                # fitness values of the children
                # must be recalculated later
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:

            # mutate an individual with probability MUTPB
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]

        fitness = list(map(toolbox.evaluate,invalid_ind))
        for ind, fit in zip(invalid_ind, fitness):
            ind.fitness.values = fit

        # The population is entirely replaced by the offspring
        pop[:] = offspring

        # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness.values[0] for ind in pop]

        hof.update(pop)
        length = len(pop)
        mean = sum(fits) / length
        sum2 = sum(x*x for x in fits)
        std = abs(sum2 / length - mean**2)**0.5

        #print("gen:",i,"  Min %s" % min(fits),"  Max %s" % max(fits),"  Avg %s" % mean,"  Std %s" % std)
        print(i,max(fits),mean)

        with open('BipedalWalker-v2.txt',mode='a') as f:
            f.write(str(i))
            f.write(" ")
            f.write(str(max(fits)))
            f.write(" ")
            f.write(str(mean))
            f.write("\n")

        if i%50 == 0:
            with open('gen'+str(i)+'checkpoint_fitness','wb') as fp:
                pickle.dump(pop,fp)

    return pop,hof
if __name__=='__main__':
    print("pop_num = ",250)
    print("gen_num ",300)
    pop,hof = main()
    #with open('gen1000fitnesshof_hardcore','wb') as fp:
    #           pickle.dump(hof,fp)
    best_ind = tools.selBest(pop,1)[0]
    for ind in hof :
        print(ind ,end="")
        print(ind.fitness)
    print(nn.update(best_ind))
