import numpy as np
from deap import algorithms,base,cma,creator,tools,benchmarks
import matplotlib.pyplot as plt
import random

class novind:
    def __init__(self,xy,novelty):
        self.xy = xy
        self.novelty = novelty

    def __repr__(self):
        return repr((self.xy,self.novelty))

class noveltymap:
    def __init__(self,k,popnum):
        print("make noveltymap")
        self.map = []
        self.k = k
        self.popnum = popnum

    def distance(self,p0,p1):
        return np.sqrt((p0[0] - p1[0])**2 + (p0[1] - p1[1]) **2)

    def insertInd(self,xy,novelty):
        self.map.append(novind(xy,novelty))

    def popInd(self,population):
        for ind in population:
            distances = np.array([self.distance(p.xy,ind) for p in self.map])
            nearest_distance = sorted(distances)[:self.k]
            novelty = sum(nearest_distance) / self.k
            self.insertInd(ind,novelty)

        return sorted(self.map,key=lambda u:u.novelty)[:self.popnum]


NGEN = 100
POPNUM = 150
N = 2
INDSIZE = N
CXPB = 0.5
MUTPB = 0.01
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
#toolbox.register("attr_float",random.random)
toolbox.register("attr_float",random.uniform,-6,6)
toolbox.register("individual",tools.initRepeat,creator.Individual,toolbox.attr_float,n=INDSIZE)
toolbox.register("population",tools.initRepeat,list,toolbox.individual)


toolbox.register("evaluate",benchmarks.himmelblau)
toolbox.register("mate",tools.cxBlend,alpha=0.5) #float
toolbox.register("mutate",tools.mutGaussian,mu=0,sigma=0.5,indpb=0.05) #mutFllipBit floatに対して津えるやつ
toolbox.register("select",tools.selTournament,tournsize=3)



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
    fitness = list(map(toolbox.evaluate,np.clip(pop,-6,6)))

    for ind, fit in zip(pop, fitness):
        ind.fitness.values = fit

    nvmap = noveltymap(5,POPNUM)

    print("gen","max","mean")
    for i in range(NGEN):
        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        print(len((offspring)))

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


        _pop = nvmap.popInd(pop)
        pop = _pop

        for ind, _ind in zip(pop,_pop):
            ind.fitness.values = _ind.novelty,
            ind  = _ind

        """
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]

        fitness = list(map(toolbox.evaluate,np.clip(invalid_ind,-6,6)))
        for ind, fit in zip(invalid_ind, fitness):
            ind.fitness.values = fit

        # The population is entirely replaced by the offspring
        pop[:] = offspring

        # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness.values[0] for ind in pop]
        """
        fits = [benchmarks.himmelblau(ind) for ind in pop]
        hof.update(pop)
        length = len(pop)
        mean = sum(fits[0]) / length
        #sum2 = sum(x*x for x in fits)
        #std = abs(sum2 / length - mean**2)**0.5
        print("gen:",i,"  Min %s" % min(fits),"  Max %s" % max(fits),"  Avg %s" % mean)
        #print("gen:",i,"  Min %s" % min(fits),"  Max %s" % max(fits),"  Avg %s" % mean,"  Std %s" % std)
        #print(i,max(fits),mean)
    return pop,hof

if __name__=='__main__':
    print("pop_num = ",POPNUM)
    print("gen_num ",NGEN)
    pop,hof = main()
    expr = tools.selBest(pop,1)[0]
    print(expr)
