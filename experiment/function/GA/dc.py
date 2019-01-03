import numpy as np
from deap import algorithms,base,cma,creator,tools,benchmarks
import matplotlib.pyplot as plt
import random

NGEN = 100
POPNUM = 150
INDSIZE = 2
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
    pop = toolbox.population(n=POPNUM)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind:ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    def func(pop):
        if pop[0] < -5 or pop[0] > 5 or pop[1] < -5 or pop[1] > 5:
            return 100000,
        pop = np.clip(pop,-5,5)
        _x = pop[0]
        _y = pop[1]
        return (1 - 1 / (1 * np.linalg.norm(np.array([_x,_y]) - [-2,-2], axis=0) + 1)) + (1 - 1 / (2 *np.linalg.norm(np.array([_x,_y]) - [4,4],  axis=0) + 1)),
    #pop,hof = algorithms.eaSimple(pop,toolbox,cxpb=0.5,mutpb=0.01,ngen=200,stats=stats,halloffame=hof,verbose=True)
    fitness = list(map(func,pop))

    for ind, fit in zip(pop, fitness):
        ind.fitness.values = fit

    print("gen ","min ","max ","mean","std")
    """
    for ind in pop:
        ax.scatter(ind[0],ind[1],c='black',marker='.')
    """
    for gen in range(NGEN):
        """
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        x = np.arange(-5, 5.05, 0.05)
        y = np.arange(-5, 5.05, 0.05)
        plt.xlim(-5,5.05)
        plt.ylim(-5,5.05)
        X ,Y= np.meshgrid(x, y)
        c1 = -2 * np.ones((2,201,201))
        c2 = 4 * np.ones((2,201,201))
        Z = (1 - 1 / (1 * np.linalg.norm(np.array([X,Y]) - c1, axis=0) + 1)) + (1 - 1 / (2 *np.linalg.norm(np.array([X,Y]) - c2,  axis=0) + 1))
        plt.pcolormesh(X, Y, Z,cmap='hsv')
        plt.colorbar () # カラーバーの表示
        plt.xlabel('X')
        plt.ylabel('Y')
        """
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
        """
        for ind in pop:
            ax.scatter(ind[0],ind[1],c='black',marker='.')

            if len(str(gen))==1:
                plt.savefig('ga_pic/00'+str(gen)+'.png')
            elif(len(str(gen))) == 2:
                plt.savefig('ga_pic/0'+str(gen)+'.png')
            elif(len(str(gen))) == 3:
                plt.savefig('ga_pic/'+str(gen)+'.png')

        """
        fitness = list(map(func,invalid_ind))
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
        print(gen  ,min(fits) ,max(fits) ,mean ,std)

        """
        if len(str(gen))==1:
            plt.savefig('ga_pic/00'+str(gen)+'.png')
        elif(len(str(gen))) == 2:
            plt.savefig('ga_pic/0'+str(gen)+'.png')
        elif(len(str(gen))) == 3:
            plt.savefig('ga_pic/'+str(gen)+'.png')
        """
        #savefig('cma_es_pic/figure'+str(gen)+'.png')

        #print("gen:",i,"  Min %s" % min(fits),"  Max %s" % max(fits),"  Avg %s" % mean,"  Std %s" % std)
        #print(i,max(fits),mean)
    #plt.show()
    return pop,hof

if __name__=='__main__':
    print("pop_num = ",POPNUM)
    print("gen_num ",NGEN)
    pop,hof = main()
    expr = tools.selBest(pop,1)[0]
    print(expr)
