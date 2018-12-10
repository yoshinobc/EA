import numpy as np
from deap import algorithms,base,cma,creator,tools,benchmarks
import matplotlib.pyplot as plt
import random
import array

IND_SIZE = 2
MIN_VALUE = -6
MAX_VALUE = -6
MIN_STRATEGY = 0.5
MAX_STRATEGY = 3

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", array.array, typecode="d",fitness=creator.FitnessMin,strategy=None)
creator.create("Strategy",array.array,typecode="d")

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

toolbox.register("evaluate",benchmarks.himmelblau)
toolbox.register("mate",tools.cxESBlend,alpha=0.5) #float
toolbox.register("mutate",tools.mutESLogNormal,c=1.0,indpb=0.3) #mutFllipBit floatに対して津えるやつ
toolbox.register("select",tools.selTournament,tournsize=3)

toolbox.decorate("mate", checkStrategy(MIN_STRATEGY))
toolbox.decorate("mutate", checkStrategy(MIN_STRATEGY))


def main():
    np.random.seed(64)
    MU, LAMBDA = 10,100
    pop = toolbox.population(n=MU)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind:ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    pop,logbook = algorithms.eaMuCommaLambda(pop,toolbox,mu=MU,lambda_=LAMBDA,cxpb=0.6,mutpb=0.3,ngen=200,stats=stats,halloffame=hof)

    return pop,logbook,hof

if __name__=='__main__':
    main()
