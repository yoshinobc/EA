
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

class hof:
    def __init__(self):
        self.hofpop = []
        self.fit = -500
    def update(self,ind,total_reward):
        self.hofpop = ind
        self.fit = total_reward

hof = hof()
env = gym.make("BipedalWalker-v2")
nn = NNN()
w_list = []
# In[9]:
count = 0
creator.create("FitnessMax",base.Fitness,weights=[1.0])
creator.create("Individual", list, fitness=creator.FitnessMax)
NGEN = 1000
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
    for _ in range(5):
        while True:
            action = get_action(observation,network)
            observation,reward,done,xylists = env.step(action)
            total_reward += reward
            if done:
                break
        if total_reward >= hof.fit:
            hof.update(individual,total_reward)
    return total_reward / 5


def EV(individual,gen):
    global count
    observation = env.reset()
    network = nn.update(individual); #networkにindividualを適用
    steps  = 0
    hull_angle_lists = []

    final_novelty = 0
    for _ in range(5):
        total_novelty = 0
        action_lists = []
        observation = env.reset()
        while True:
            action = get_action(observation,network)
            observation,reward,done,xylists = env.step(action)
            action_lists.append(xylists)
        #if (-4.8  > observation[0]) or (observation[0] > 4.8) or (0.017453292519943 < observation[3] < -0.017453292519943) or (episode_reward >= MAX_STEPS):
            if done:
                break
        for i in range(len(action_lists) - 1):
            total_novelty += np.sqrt((action_lists[i+1][0] - action_lists[i][0])**2 + (action_lists[i+1][1] - action_lists[i][1])**2)
        final_novelty += total_novelty
    return final_novelty / 2,

def EV2(individual):
    observation = env.reset()
    total_fitness = 0
    network = nn.update(individual)
    for _ in range(2):
        observation = env.reset()
        while True:
            action = get_action(observation,network)
            observation,reward,done,xylists = env.step(action)
            if done:
                break
            total_fitness += reward
    return total_fitness / 2,
toolbox.register("evaluate",EV,gen = 0)
toolbox.register("evaluate_fit",EV2)
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
    #if (-4.8  > observation[0]) or (observation[0] > 4.8) or (0.017453292519943 < observation[3] < -0.017453292519943) or (episode_reward >= MAX_STEPS):
        if done:
            ENV.render()
            #if total_reward >= 280:
                #print(individial)
            break

def main():
    global max_fitness
    global mean_fitness

    random.seed(1)

    pop = toolbox.population(n=250)
    """
    with open('gen100checkpoints', 'rb') as f:
        pop = pickle.load(f)
    """
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
    print("gen","max","mean")
    flag = 0
    max_fits = 0
    for i in range(NGEN):
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

        if flag == 0:
            fitnesses = map(toolbox.evaluate_fit,invalid_ind)
            for ind,fit in zip(invalid_ind,fitnesses):
                ind.fitness.values = fit

            pop[:] = offspring

            # Gather all the fitnesses in one list and print the stats


        else:
            print("novelty")
            flag = 0
            novelties = map(toolbox.evaluate, invalid_ind)
            for ind, nov in zip(invalid_ind, novelties):
                ind.fitness.values = nov

            offspring.sort(key = operator.attrgetter('fitness.values'))
            """
            for i, ind in enumerate(pop):
                print(i, ind.fitness.values)
            """
            popmean = offspring[75]
            for ind in offspring:
                ind.fitness.values = (ind.fitness.values[0] - popmean.fitness.values[0])**2,

            # The population is entirely replaced by the offspring
            pop[:] = offspring

        fits = [getfit(ind) for ind in pop]
        
        length = len(pop)
        mean = sum(fits) / length
        sum2 = sum(x*x for x in fits)
        std = abs(sum2 / length - mean**2)**0.5
        if abs(max(fits) - max_fits) <= 1:
            #print("flags")
            frag = 1
        #print(max(fits),max_fits,max(fits) - max_fits)
        max_fits = max(fits)
        #print("gen:",i,"  Min %s" % min(fits),"  Max %s" % max(fits),"  Avg %s" % mean,"  Std %s" % std)

        print(i,max(fits),mean)
        with open('liner_novelty.txt',mode='a') as f:
            f.write(str(i))
            f.write(" ")
            f.write(str(max(fits)))
            f.write(" ")
            f.write(str(mean))
            f.write("\n")
        if i%50 == 0:
            with open('gen'+str(i)+'checkpoint_maxstepsliner_hardcore','wb') as fp:
                pickle.dump(pop,fp)

            #print(hof.fit,hof.hofpop)

    """
    for ind in pop:
        rendering(ind)
    """
    return pop,hof
if __name__=='__main__':
    pop,hof = main()
    with open('gen1000hof','wb') as fp:
                pickle.dump(hof,fp)
    print(hof.fit,hof.hofpop)
