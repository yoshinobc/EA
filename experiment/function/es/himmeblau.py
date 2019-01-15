import numpy as np
from deap import algorithms,base,cma,creator,tools,benchmarks
import matplotlib.pyplot as plt
import random
import array
from deap.algorithms import varOr
import time

IND_SIZE = 2
MIN_VALUE = -5
MAX_VALUE = -5
MIN_STRATEGY = 0.5
MAX_STRATEGY = 3
NGEN = 200

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", array.array, typecode="d",fitness=creator.FitnessMin,strategy=None)
creator.create("Strategy",array.array,typecode="d")

def func(pop):
    return (1 - 1 / (1 * np.linalg.norm(np.array(pop) - [-2,-2], axis=0) + 1)) + (1 - 1 / (2 *np.linalg.norm(np.array(pop) - [4,4],  axis=0) + 1)),

def generateES(icls, scls, size, imin, imax, smin, smax):
    ind = icls(random.uniform(imin, imax) for _ in range(size))
    ind.strategy = scls(random.uniform(smin, smax) for _ in range(size))
    return ind

def checkStrategy(minstrategy):
    def decorator(func):
        def wrappper(*args, **kargs):
            children = func(*args, **kargs)
            for child in children:
                for i, s in enumerate(child.strategy):
                    if s < minstrategy:
                        child.strategy[i] = minstrategy
            return children
        return wrappper
    return decorator


toolbox = base.Toolbox()
toolbox.register("individual",generateES,creator.Individual,creator.Strategy,IND_SIZE,MIN_VALUE,MAX_VALUE,MIN_STRATEGY,MAX_STRATEGY)
toolbox.register("population",tools.initRepeat,list,toolbox.individual)

toolbox.register("evaluate",func)
toolbox.register("mate",tools.cxESBlend,alpha=0.1) #float
toolbox.register("mutate",tools.mutESLogNormal,c=1.0,indpb=0.03) #mutFllipBit floatに対して津えるやつ
toolbox.register("select",tools.selTournament,tournsize=3)

toolbox.decorate("mate", checkStrategy(MIN_STRATEGY))
toolbox.decorate("mutate", checkStrategy(MIN_STRATEGY))
"""
def varOr(population, toolbox, lambda_, cxpb, mutpb):
    assert (cxpb + mutpb) <= 1.0, (
        "The sum of the crossover and mutation probabilities must be smaller "
        "or equal to 1.0.")
    offspring = []
    for _ in range(lambda_):
        op_choice = random.random()
        if op_choice < cxpb:            # Apply crossover
            ind1, ind2 = map(toolbox.clone, random.sample(population, 2))
            ind1, ind2 = toolbox.mate(ind1, ind2)
            del ind1.fitness.values
            offspring.append(ind1)
        elif op_choice < cxpb + mutpb:  # Apply mutation
            ind = toolbox.clone(random.choice(population))
            ind, = toolbox.mutate(ind)
            del ind.fitness.values
            offspring.append(ind)
        else:                           # Apply reproduction
            offspring.append(random.choice(population))

    return offspring
"""
def main():
    np.random.seed(64)
    MU, LAMBDA = 15,150
    population = toolbox.population(n=MU)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    """
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit
    """
    stop_gen = 200
    ok_count = 0
    """
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)

    for gen in range(NGEN):
        offspring = varOr(population, toolbox, LAMBDA, cxpb = 0.6, mutpb = 0.3)
        invalid_ind = [ind for ind in population if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        if halloffame is not None:
            halloffame.update(offspring)
        fits = [ind.fitness.values[0] for ind in population]
        length = len(population)
        mean = sum(fits) / length
        sum2 = sum(x*x for x in fits)
        std = abs(sum2 / length - mean**2)**0.5
        print(gen ,min(fits) ,max(fits) ,mean,std)

        if min(fits) <= np.exp(-10) :
            ok_count = 1
            stop_gen = gen
            break

        population = toolbox.select(offspring, MU)
    """
    pop,logbook,stop_gen,ok_count = algorithms.eaMuCommaLambda(population,toolbox,mu=MU,lambda_=LAMBDA,cxpb=0.6,mutpb=0.3,ngen=200,stats=stats,halloffame=hof)

    return population,stop_gen,ok_count

if __name__=='__main__':
    count = 0
    sum_gen = 0
    start = time.time()
    for i in range(500):
        pop,stop_gen,ok_count = main()
        count += ok_count
        sum_gen += stop_gen
        print(i,ok_count,pop)
    episode_time = time.time() - start
    print("count",count)
    print("sum_gen",sum_gen / 1000)
    print("time",episode_time)
