
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
import operator
import pickle

env = gym.make("BipedalWalker-v2")
nn = NNN()
MAX_STEPS = 2000
w_list = []
# In[9]:
count = 0
creator.create("FitnessMax",base.Fitness,weights=[1.0])
creator.create("Individual", list, fitness=creator.FitnessMax)
NGEN = 500
CXPB=0.5
MUTPB=0.2
toolbox = base.Toolbox()
max_fitness = 0
mean_fitness = 0
def decide_weight(network):
    #print(np.reshape(network['W1'],(12)))
    w_list = []
    if not network :
        w_list.extend(np.reshape(np.random.normal(0,1,(24,16)),(384)))
        w_list.extend(np.reshape(np.random.normal(0,1,(1,16)),(16)))
        w_list.extend(np.reshape(np.random.normal(0,1,(16,8)),(128)))
        w_list.extend(np.reshape(np.random.normal(0,1,(1,8)),(8)))
        w_list.extend(np.reshape(np.random.normal(0,1,(8,4)),(32)))
        w_list.extend(np.reshape(np.random.normal(0,1,(1,4)),(4)))

    return w_list

toolbox.register("gene",decide_weight,nn.network)
toolbox.register("individual",tools.initIterate,creator.Individual,toolbox.gene)
toolbox.register("population",tools.initRepeat,list,toolbox.individual)

def getfit(individual):
    observation = env.reset()
    network = nn.update(individual)
    total_reward = 0
    for i in range(3700):
        action = get_action(observation,network)
        observation,reward,done,xylists = env.step(action)
        total_reward += reward
        if done:
            break
    return total_reward


def EV(individual,gen):
    global count
    observation = env.reset()
    network = nn.update(individual); #networkにindividualを適用
    observatin = env.reset()
    total_reward = 0
    steps  = 0
    action_lists = []
    hull_angle_lists = []

    if gen <= 450:
        MAX_STEPS = 3700
    elif gen <= 400:
        MAX_STEPS = 3000
    elif gen <= 350:
        MAX_STEPS = 2500
    elif gen <= 300:
        MAX_STEPS = 1800
    elif gen <= 250:
        MAX_STEPS = 1200
    elif gen <= 200:
        MAX_STEPS = 800
    elif gen <= 150:
        MAX_STEPS = 500
    elif gen <= 100:
        MAX_STEPS = 300
    elif gen <= 50:
        MAX_STEPS = 100

    for i in range(MAX_STEPS):
        global max_fitness
        global mean_fitness

        #env = wrappers.Monitor(env,'/mnt/c/Users/bc/Documents/EA/BipedalWalker/movies/',force=True)
        action = get_action(observation,network)
        observation,reward,done,xylists = env.step(action)
        action_lists.append(xylists)
        hull_angle_lists.append(observation[0])
        total_reward += reward
        #if (-4.8  > observation[0]) or (observation[0] > 4.8) or (0.017453292519943 < observation[3] < -0.017453292519943) or (episode_reward >= MAX_STEPS):
        steps = i
        if done:
            break
    if total_reward >= 280:
            print(individual)
    total_novelty = 0
    for i in range(len(action_lists) - 1):
        total_novelty +=np.sqrt((action_lists[i+1][0] - action_lists[i][0])**2 + (action_lists[i+1][1] - action_lists[i][1])**2)
    return total_novelty,

toolbox.register("evaluate",EV,gen = 0)
toolbox.register("mate",tools.cxBlend,alpha=0.5) #float
toolbox.register("mutate",tools.mutGaussian,mu=0,sigma=0.5,indpb=0.05) #mutFllipBit floatに対して津えるやつ
toolbox.register("select",tools.selTournament,tournsize=3)

def get_action(observation,network):
    action = decide_action(observation,network)
    return action

def decide_action(observation,network):
    return nn.conclusion(observation,network)

def rendering(individual):
    global env
    network = nn.update(individual); #networkにindividualを適用
    total_reward = 0
    while True:
        ENV = wrappers.Monitor(env,'/mnt/c/Users/bc/Documents/EA/BipedalWalker/noveltymovies/',force=True)
        observation = ENV.reset()
        action = get_action(observation,network)
        observation,reward,done,xylists = ENV.step(action)
        total_reward += reward
        ENV.render()
    #if (-4.8  > observation[0]) or (observation[0] > 4.8) or (0.017453292519943 < observation[3] < -0.017453292519943) or (episode_reward >= MAX_STEPS):
        if done:
            ENV.render()
            if total_reward >= 280:
                print(individial)
            break
def main():
    global max_fitness
    global mean_fitness

    random.seed(1)

    pop = toolbox.population(n=150)
    with open('gen100checkpoints', 'rb') as f:
        pop = pickle.load(f)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind:ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    #pop,log = algorithms.eaSimple(pop,toolbox,cxpb=0.5,mutpb=0.2,ngen=200,stats=stats,halloffame=hof,verbose=True)
    novelties = list(map(toolbox.evaluate,pop))
    for ind, nov in zip(pop, novelties):
        ind.fitness.values = nov
    pop.sort(key = operator.attrgetter('fitness.values'))
    popmean = pop[75]
    for ind in pop:
        ind.fitness.values = (ind.fitness.values[0] - popmean.fitness.values[0])**2,
    fits = [getfit(ind) for ind in pop]
    for i in range(100,NGEN+1):
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
        novelties = map(toolbox.evaluate, invalid_ind)
        for ind, nov in zip(invalid_ind, novelties):
            ind.fitness.values = nov
        pop.sort(key = operator.attrgetter('fitness.values'))
        """
        for i, ind in enumerate(pop):
            print(i, ind.fitness.values)
        """
        popmean = pop[75]
        for ind in pop:
            ind.fitness.values = (ind.fitness.values[0] - popmean.fitness.values[0])**2,

        # The population is entirely replaced by the offspring
        pop[:] = offspring

        # Gather all the fitnesses in one list and print the stats
        fits = [getfit(ind) for ind in pop]

        length = len(pop)
        mean = sum(fits) / length
        sum2 = sum(x*x for x in fits)
        std = abs(sum2 / length - mean**2)**0.5

        print("gen:",i,"  Min %s" % min(fits),"  Max %s" % max(fits),"  Avg %s" % mean,"  Std %s" % std)
        if i%50 == 0:
            with open('gen'+str(i)+'checkpoint','wb') as fp:
                pickle.dump(pop,fp)
    for ind in pop:
        rendering(ind)
    return pop,hof
if __name__=='__main__':
    pop,hof = main()
