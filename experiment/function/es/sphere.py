import array
import random

import numpy

from deap import algorithms
from deap import base
from deap import benchmarks
from deap import creator
from deap import tools

IND_SIZE = 2
MIN_VALUE = -10
MAX_VALUE = 10
MIN_STRATEGY = 0.5
MAX_STRATEGY = 3

def func(pop):

      return (1 - 1 / (1 * numpy.linalg.norm(numpy.array(pop) - [-2,-2], axis=0) + 1)) + (1 - 1 / (2 *numpy.linalg.norm(numpy.array(pop) - [4,4],  axis=0) + 1)),


creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", array.array, typecode="d", fitness=creator.FitnessMin, strategy=None)
creator.create("Strategy", array.array, typecode="d")

# Individual generator
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
toolbox.register("individual", generateES, creator.Individual, creator.Strategy,
    IND_SIZE, MIN_VALUE, MAX_VALUE, MIN_STRATEGY, MAX_STRATEGY)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxESBlend, alpha=0.1)
toolbox.register("mutate", tools.mutESLogNormal, c=1.0, indpb=0.03)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", func)

toolbox.decorate("mate", checkStrategy(MIN_STRATEGY))
toolbox.decorate("mutate", checkStrategy(MIN_STRATEGY))

def main():
    random.seed()
    MU, LAMBDA = 15, 150
    pop = toolbox.population(n=MU)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)

    pop, logbook,stop_gen,ok_count = algorithms.eaMuCommaLambda(pop, toolbox, mu=MU, lambda_=LAMBDA,
        cxpb=0.6, mutpb=0.3, ngen=200, stats=stats, halloffame=hof)
    gen,fitmin = logbook.select("gen","min")
    for i,fit in enumerate(fitmin):
        print(i, fit)
    print(pop)
    return pop, logbook, hof

if __name__ == "__main__":
    main()
