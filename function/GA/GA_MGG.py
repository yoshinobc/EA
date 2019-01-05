import numpy as np
from deap import algorithms,base,cma,creator,tools,benchmarks
import operator
import matplotlib.pyplot as plt
import random

NGEN = 100
POPNUM = 150
INDSIZE = 2
CXPB = 0.5
MUTPB = 0.01
C = 20

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
#toolbox.register("attr_float",random.random)
toolbox.register("attr_float",random.uniform,-6,6)
toolbox.register("individual",tools.initRepeat,creator.Individual,toolbox.attr_float,n=INDSIZE)
toolbox.register("population",tools.initRepeat,list,toolbox.individual)

def selMGG(pop,POPNUM):
    offspring = []
    for _ in range(int(POPNUM / 2)):
        parchild = []
        if random.random() < 1:
            parent = toolbox.selectRandom(pop)
            parchild.append(parent[0])
            parchild.append(parent[1])
            for _ in range(C):
                cp_parent0 = parent[0]
                cp_parent1 = parent[1]
                toolbox.mate(cp_parent0,cp_parent1)
                parchild.append(cp_parent0)
                parchild.append(cp_parent1)
                del cp_parent0.fitness.values
                del cp_parent1.fitness.values
        fitness = list(map(toolbox.evaluate,parchild))
        for ind, fit in zip(parchild, fitness):
            ind.fitness.values = fit

        parchild.sort(key = operator.attrgetter('fitness.values'))
        offspring.append(parchild[0])
        offspring.append(toolbox.selectRoulette(parchild)[0])
    return offspring

toolbox.register("evaluate",benchmarks.sphere)
toolbox.register("mate",tools.cxBlend,alpha=0.5) #float
toolbox.register("mutate",tools.mutGaussian,mu=0,sigma=0.5,indpb=0.02) #mutFllipBit floatに対して津えるやつ
toolbox.register("selectMGG",selMGG)
toolbox.register("selectRandom",tools.selRandom,k=2)
toolbox.register("selectBest",tools.selBest,k=1)
toolbox.register("selectRoulette",tools.selRoulette,k=1)


def main():
    np.random.seed(64)
    pop = toolbox.population(n=POPNUM)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind:ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    #pop,hof = algorithms.eaSimple(pop,toolbox,cxpb=0.5,mutpb=0.01,ngen=200,stats=stats,halloffame=hof,verbose=True)
    fitness = list(map(toolbox.evaluate,pop))
    for ind, fit in zip(pop, fitness):
        ind.fitness.values = fit

    print("gen ","min ","max ","mean")
    """
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    for ind in pop:
        ax.scatter(ind[0],ind[1],c='blue',marker='.')
    """
    for gen in range(NGEN):
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        # Select the next generation individuals
        offspring = toolbox.selectMGG(pop,len(pop))

        offspring = list(map(toolbox.clone, offspring))
        for mutant in offspring:

            # mutate an individual with probability MUTPB
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
        # Evaluate the individuals with an invalid fitness
        """
        for ind in offspring2:
            ax.scatter(ind[0],ind[1],c='blue',marker='.')
        """
        fitness = list(map(toolbox.evaluate,offspring))
        for ind, fit in zip(offspring, fitness):
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
        print(gen ,min(fits) ,max(fits) ,mean)
        """
        if len(str(gen))==1:
            plt.savefig('GA_MGG/00'+str(gen)+'.png')
        elif(len(str(gen))) == 2:
            plt.savefig('GA_MGG/0'+str(gen)+'.png')
        elif(len(str(gen))) == 3:
            plt.savefig('GA_MGG/'+str(gen)+'.png')
        #savefig('cma_es_pic/figure'+str(gen)+'.png')
        """
        if min(fits) == 0:
            break
        #print("gen:",i,"  Min %s" % min(fits),"  Max %s" % max(fits),"  Avg %s" % mean,"  Std %s" % std)
        #print(i,max(fits),mean)
    return pop,hof

if __name__=='__main__':
    print("pop_num = ",POPNUM)
    print("gen_num ",NGEN)
    pop,hof = main()
    print(hof)
    expr = tools.selBest(pop,1)[0]
    print(expr,expr.fitness)
