import numpy as np
from deap import algorithms,base,cma,creator,tools,benchmarks
import matplotlib.pyplot as plt
import random
import csv
import time

NGEN = 200
POPNUM = 150
INDSIZE = 7
CXPB = 0.5
MUTPB = 0.01
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
#toolbox.register("attr_float",random.random)
toolbox.register("attr_float",random.uniform,-5,5)
toolbox.register("individual",tools.initRepeat,creator.Individual,toolbox.attr_float,n=INDSIZE)
toolbox.register("population",tools.initRepeat,list,toolbox.individual)


toolbox.register("evaluate",benchmarks.griewank)
toolbox.register("mate",tools.cxBlend,alpha=0.5) #float
toolbox.register("mutate",tools.mutGaussian,mu=0,sigma=0.5,indpb=0.05) #mutFllipBit floatに対して津えるやつ
toolbox.register("select",tools.selTournament,tournsize=3)



def main():

    np.random.seed(64)
    pop = toolbox.population(n=POPNUM)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind:ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    ok_count = 0

    #pop,hof = algorithms.eaSimple(pop,toolbox,cxpb=0.5,mutpb=0.01,ngen=200,stats=stats,halloffame=hof,verbose=True)
    fitness = list(map(toolbox.evaluate,pop))

    for ind, fit in zip(pop, fitness):
        ind.fitness.values = fit

    #print("gen ","min ","max ","mean","std")
    stop_gen = 200
    for gen in range(NGEN):
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
        #print(gen  ,min(fits) ,max(fits) ,mean ,std)

        with open('griewank.txt',mode='a') as f:
            f.write(str(gen))
            f.write(" ")
            f.write(str(min(fits)))
            f.write(" ")
            f.write(str(mean))
            f.write("\n")
        """
        time 20.42633295059204
        [-3.191757675565447, 2.158503392015575e-09, 16.294035978157076, -6.375366880774475, -7.618792813179073, 1.1405896238647264e-08, 0.00027775284370455254]
        """
        if min(fits) <= np.exp(-10):
            stop_gen = gen
            ok_count = 1
            break
        
    return pop,hof,ok_count,stop_gen

if __name__=='__main__':
    print("pop_num = ",POPNUM)
    print("gen_num ",NGEN)
    count = 0
    trials = 500
    count_gen = 0
    start = time.time()
    for i in range(500):
        pop,hof,ok_count,stop_gen = main()
        count += ok_count
        count_gen += stop_gen
        print(i,ok_count,hof)
    etime = time.time() - start
    print("count",count)

    print("stop_gen",count_gen / 500)
    print("time",etime)
    expr = tools.selBest(pop,1)[0]
    print(expr)
    """
    count 1
    stop_gen 199.724
    time 357.9312949180603
    [-15.700360170374521, -0.0003372959712526326, -10.866424683827812, 12.644161731985704, 0.00067266294532439, -0.0011384235858281897, -8.283019320863794]
    """

