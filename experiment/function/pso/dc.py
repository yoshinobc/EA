import operator
import random

import numpy as np
import time
from deap import base
from deap import benchmarks
from deap import creator
from deap import tools

def func(pop):
      """
      if pop[0] < -5 or pop[0] > 5 or pop[1] < -5 or pop[1] > 5:
          return 100000,
      pop = np.clip(pop,-5,5)
      """
      _x = pop[0]
      _y = pop[1]
      return (1 - 1 / (1 * np.linalg.norm(np.array(pop) - [-2,-2,-2,-2,-2,-2,-2], axis=0) + 1)) + (1 - 1 / (2 *np.linalg.norm(np.array(pop) - [4,4,4,4,4,4,4],  axis=0) + 1)),

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Particle", list, fitness=creator.FitnessMin, speed=list,
    smin=None, smax=None, best=None)

def generate(size, pmin, pmax, smin, smax):
    part = creator.Particle(random.uniform(pmin, pmax) for _ in range(size))
    part.speed = [random.uniform(smin, smax) for _ in range(size)]
    part.smin = smin
    part.smax = smax
    return part

def updateParticle(part, best, phi1, phi2):
    u1 = (random.uniform(0, phi1) for _ in range(len(part)))
    u2 = (random.uniform(0, phi2) for _ in range(len(part)))
    v_u1 = map(operator.mul, u1, map(operator.sub, part.best, part))
    v_u2 = map(operator.mul, u2, map(operator.sub, best, part))
    part.speed = list(map(operator.add, part.speed, map(operator.add, v_u1, v_u2)))
    for i, speed in enumerate(part.speed):
        if speed < part.smin:
            part.speed[i] = part.smin
        elif speed > part.smax:
            part.speed[i] = part.smax
    part[:] = list(map(operator.add, part, part.speed))

toolbox = base.Toolbox()
toolbox.register("particle", generate, size=7, pmin=-5, pmax=5, smin=-3, smax=3)
toolbox.register("population", tools.initRepeat, list, toolbox.particle)
toolbox.register("update", updateParticle, phi1=2.0, phi2=2.0)
#toolbox.register("evaluate", benchmarks.shekel(a = [[-2,-2,-2,-2,-2,-2,-2],[4,4,4,4,4,4,4]],c = [0.002,0.005]))

def main():
    pop = toolbox.population(n=150)
    stats = tools.Statistics(lambda ind: ind.fitness.values)


    logbook = tools.Logbook()
    logbook.header = ["gen", "evals"] + stats.fields

    GEN = 200
    best = None
    stop_gen = 200
    ok_count = 0
    for gen in range(GEN):

        for part in pop:
            part.fitness.values = func(part)
            if not part.best or part.best.fitness < part.fitness:
                part.best = creator.Particle(part)
                part.best.fitness.values = part.fitness.values
            if not best or best.fitness < part.fitness:
                best = creator.Particle(part)
                best.fitness.values = part.fitness.values

        fitnesses = toolbox.map(func, pop)
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit

        fits = [ind.fitness.values[0] for ind in pop]

        length = len(pop)
        mean = sum(fits) / length
        sum2 = sum(x*x for x in fits)
        std = abs(sum2 / length - mean**2)**0.5
        if min(fits) <= 0.9407390124845117 + np.exp(-10) :
            stop_gen = gen
            ok_count = 1
            break

        for part in pop:
            toolbox.update(part, best)

        # Gather all the fitnesses in one list and print the stats
        #gbook.record(gen=g, evals=len(pop), **stats.compile(pop))
        #print(logbook.stream)

    return pop, logbook, best,stop_gen,ok_count

if __name__ == "__main__":
    count = 0
    trials = 500
    count_gen = 0
    start = time.time()
    for i in range(trials):
        pop,logbook,best,stop_gen,ok_count = main()
        print(i,ok_count)
        count += ok_count
        count_gen += stop_gen
    etime = time.time() - start
    print("count",count)
    print("stop_gen",count_gen / trials)
    print("time",etime)
"""
count 500
stop_gen 37.512
time 102.77915549278259
"""