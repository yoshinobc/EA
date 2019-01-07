import numpy

from deap import algorithms
from deap import base
from deap import benchmarks
from deap import cma
from deap import creator
from deap import tools
import random
import matplotlib.pyplot as plt

N = 2  # 問題の次元
INDSIZE = 2
POPNUM = 150
NGEN = 100  # 総ステップ数
CXPB = 0.5
MUTPB = 0.2

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("attr_float",random.uniform,-6,6)
toolbox.register("individual",tools.initRepeat,creator.Individual,toolbox.attr_float,n=INDSIZE)
toolbox.register("population",tools.initRepeat,list,toolbox.individual)

toolbox.register("mate",tools.cxBlend,alpha=0.5) #float
toolbox.register("mutate",tools.mutGaussian,mu=0,sigma=0.5,indpb=0.05) #mutFllipBit floatに対して津えるやつ
toolbox.register("select",tools.selTournament,tournsize = 3)

def mapping(gen):
    if gen <= 22:
        num = gen
    elif 22 < gen <= 44:
        num = 44 - gen
    elif 44 < gen <= 66:
        num = gen - 45
    elif 66 < gen <= 88:
        num = 88 - gen
    elif 88 < gen:
        num = gen - 89

    return num


def main():
    numpy.random.seed(64)
    population = toolbox.population(n=POPNUM)
    hof = tools.HallOfFame(1)


    halloffame = tools.HallOfFame(1)

    x = numpy.arange(-6, 6, 0.05) #x軸の描画範囲の生成。0から10まで0.05刻み。
    y = numpy.arange(-6, 6, 0.05) #y軸の描画範囲の生成。0から10まで0.05刻み。

    X, Y = numpy.meshgrid(x, y)

    print("gen ","min ","max ","mean ","std")
    for gen in range(NGEN):
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.set_xlim([-6,6])
        ax.set_ylim([-6,6])
        Z = numpy.power((numpy.power(X,2) + Y - mapping(gen)),2) + numpy.power((X + numpy.power(Y,2) - mapping(gen)),2)
        def func(pop):
            pop = numpy.clip(pop,-6,6)
            X = pop[0]
            Y = pop[1]
            return numpy.power((numpy.power(X,2) + Y - mapping(gen)),2) + numpy.power((X + numpy.power(Y,2) - mapping(gen)),2)   # 表示する計算式の指定。等高線はZに対して作られる。
        #Z = (1 - 1/(1 + 0.05 * (np.power(X,2)+(np.power((Y - 10),2)))) - 1/(1 + 0.05*(np.power((X - 10),2) + np.power(Y,2))) - 1/(1 + 0.03*(np.power((X + 10),2) + np.power(Y,2))) - 1/(1 + 0.05*(np.power((X - 5),2) + np.power((Y + 10),2))) - 1/(1 + 0.1*(np.power((X + 5),2) + np.power((Y + 10),2))))*(1 + 0.0001*np.power((np.power(X,2) + np.power(Y,2)),1.2))
        #plt.pcolormesh(X, Y, Z, cmap='hsv') # 等高線図の生成。cmapで色付けの規則を指定する。
        plt.pcolormesh(X, Y, Z,cmap='hsv') # 等高線図の生成。cmapで色付けの規則を指定す
        #pp=plt.colorbar (orientation="vertical") # カラーバーの表示
        #pp.set_label("Label", fontname="Arial", fontsize=24) #カラーバーのラベル

        plt.xlabel('X', fontsize=24)
        plt.ylabel('Y', fontsize=24)

        for ind in population:
            ax.scatter(ind[0],ind[1],c='blue',marker='.')

        offspring = toolbox.select(population,len(population))
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

        # The population is entirely replaced by the offspring
        population[:] = offspring

        # Evaluate the individuals with an invalid fitness
        for ind in population:
            ax.scatter(ind[0],ind[1],c='blue',marker='.')


        fitness = list(map(func,numpy.clip(population,-6,6)))
        for ind, fit in zip(offspring, fitness):
            ind.fitness.values = fit,


        # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness.values[0] for ind in population]

        hof.update(population)


        length = len(population)
        mean = sum(fits) / length
        sum2 = sum(x*x for x in fits)
        std = abs(sum2 / length - mean**2)**0.5
        print(gen ,min(fits) ,max(fits) ,mean,std)

        if len(str(gen))==1:
            plt.savefig('ga_pic_mutpb/00'+str(gen)+'.png')
        elif(len(str(gen))) == 2:
            plt.savefig('ga_pic_mutpb/0'+str(gen)+'.png')
        elif(len(str(gen))) == 3:
            plt.savefig('ga_pic_mutpb/'+str(gen)+'.png')
        #savefig('cma_es_pic/figure'+str(gen)+'.png')
        plt.clf()
        plt.close()
        if min(fits) == 0:
            break
    # 計算結果を描画
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from matplotlib.patches import Ellipse
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    X = numpy.arange(-6, 6, 0.1)
    Y = numpy.arange(-6, 6, 0.1)
    X, Y = numpy.meshgrid(X, Y)
    Z = [[benchmarks.rastrigin((x, y))[0] for x, y in zip(xx, yy)]
         for xx, yy in zip(X, Y)]
    ax.imshow(Z, cmap=cm.jet, extent=[-5.12, 5.12, -5.12, 5.12])
    for x, sigma, xmean in zip(halloffame_array, C_array, centroid_array):
        # 多変量分布の分散を楕円で描画
        Darray, Bmat = numpy.linalg.eigh(sigma)
        ax.add_artist(Ellipse((xmean[0], xmean[1]),
                              numpy.sqrt(Darray[0]),
                              numpy.sqrt(Darray[1]),
                              numpy.arctan2(Bmat[1, 0], Bmat[0, 0]) * 180 / numpy.pi,
                              color="g",
                              alpha=0.7))
        ax.plot([x[0]], [x[1]], c='r', marker='o')
        ax.axis([-6, 6, -6, 6])
        plt.draw()
    plt.show(block=True)
    """

if __name__ == "__main__":
    main()
