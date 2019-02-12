import numpy as np
from deap import algorithms,base,cma,creator,tools,benchmarks
import matplotlib.pyplot as plt
import random
import time

class novind:
    def __init__(self,xy,novelty,fitness):
        self.xy = xy
        self.novelty = novelty
        self.fitness = fitness

    def __repr__(self):
        return repr((self.xy,self.novelty))

class noveltymap:
    def __init__(self,k,popnum):
        #print("make noveltymap")
        self.map = []
        self.k = k
        self.popnum = popnum
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(1,1,1)

    def distance(self,p0,p1):
        return np.sqrt((p0[0] - p1[0])**2 + (p0[1] - p1[1]) **2)

    def insertInd(self,xy,novelty,fitness):
        self.map.append(novind(xy,novelty,fitness))
        #self.ax.scatter(xy[0],xy[1],c='red',marker='.')

    def calFit(self,ind):
        return toolbox.evaluate(ind)[0]

    def popInd(self,population):
        for ind in population:
            distances = np.array([self.distance(p.xy,ind) for p in self.map])
            nearest_distance = sorted(distances)[:self.k]
            novelty = sum(nearest_distance) / self.k
            fitness = self.calFit(ind)
            self.insertInd(ind,novelty,fitness)
            #self.insertInd(ind,novelty)
        return sorted(self.map,key=lambda u:u.novelty)[:self.popnum]


NGEN = 200
POPNUM = 150
N = 7
INDSIZE = N
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

def func(pop):
      return (1 - 1 / (1 * np.linalg.norm(np.array(pop) - [-2,-2,-2,-2,-2,-2,-2], axis=0) + 1)) + (1 - 1 / (2 *np.linalg.norm(np.array(pop) - [4,4,4,4,4,4,4],  axis=0) + 1)),

def func(pop):
      return (1 - 1 / (1 * np.linalg.norm(np.array(pop) - [-2,-2], axis=0) + 1)) + (1 - 1 / (2 *np.linalg.norm(np.array(pop) - [4,4],  axis=0) + 1)),


def main():
    np.random.seed(64)
    pop = toolbox.population(n=150)
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
    min_fit = 0
    nvmap = noveltymap(5,POPNUM)
    step = 0
    #print("gen ","min ""max ","mean")
    novflag = 0
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
        if novflag == 2:
            #print("nov")
            novflag = 0
            _pop = nvmap.popInd(pop)
            for off ,_ind in zip(offspring,_pop):
                off = _ind.xy
                off.fitness.values = _ind.novelty,
                pop[:] = offspring
                fits = [benchmarks.himmelblau(ind)[0] for ind in pop]
        else:
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            """
            for ind in invalid_ind:
                nvmap.ax.scatter(ind[0],ind[1],c='blue',marker='.')
            """
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
        #sum2 = sum(x*x for x in fits)
        #std = abs(sum2 / length - mean**2)**0.5
        if abs(min(fits) - min_fit) == 0.001:
            novflag += 1
        else :
            novflag = 0
        #print(i ,min(fits) ,max(fits) ,mean)
        min_fit = min(fits)
        if min(fits) <= np.exp(-10):
            step +=1
            break
        #print("gen:",i,"  Min %s" % min(fits),"  Max %s" % max(fits),"  Avg %s" % mean)
        #print("gen:",i,"  Min %s" % min(fits),"  Max %s" % max(fits),"  Avg %s" % mean,"  Std %s" % std)
        #print(i,max(fits),mean)

    #nvmap.fig.show()
    #time.sleep(100000000)
    return pop,hof,step

if __name__=='__main__':
    print("pop_num = ",POPNUM)
    print("gen_num ",NGEN)
    total = 0
    for i in range(500):
        pop,hof,step = main()
        total += step
        print(i,step,hof)
    print(total)
    expr = tools.selBest(pop,1)[0]
    print(expr)
"""
2回連続で最小値が一致していたらnovelty_searchを行う．
"""