import numpy

from deap import algorithms
from deap import base
from deap import benchmarks
from deap import cma
from deap import creator
from deap import tools

import matplotlib.pyplot as plt

N = 2  # 問題の次元
NGEN = 300  # 総ステップ数

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("evaluate", benchmarks.himmelblau)

def main():
    numpy.random.seed(64)

    # The CMA-ES algorithm
    strategy = cma.Strategy(centroid=[5.0]*N, sigma=3.0, lambda_=20*N)
    toolbox.register("generate", strategy.generate, creator.Individual)
    toolbox.register("update", strategy.update)

    halloffame = tools.HallOfFame(1)

    halloffame_array = []
    C_array = []
    centroid_array = []
    print("gen ","min ","max ","mean ","std")
    for gen in range(NGEN):
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        # 新たな世代の個体群を生成
        population = toolbox.generate()
        # 個体群の評価

        for ind in population:
            ax.scatter(ind[0],ind[1],c='blue',marker='.')

        fitnesses = toolbox.map(toolbox.evaluate, population)
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit

        fits = [ind.fitness.values[0] for ind in population]

        length = len(population)
        mean = sum(fits) / length
        sum2 = sum(x*x for x in fits)
        std = abs(sum2 / length - mean**2)**0.5
        print(gen ,min(fits) ,max(fits) ,mean,std)

        # 個体群の評価から次世代の計算のためのパラメタ更新
        toolbox.update(population)

        # hall-of-fameの更新
        halloffame.update(population)

        halloffame_array.append(halloffame[0])
        C_array.append(strategy.C)
        centroid_array.append(strategy.centroid)
        if len(str(gen))==1:
            plt.savefig('cma_es_pic/00'+str(gen)+'.png')
        elif(len(str(gen))) == 2:
            plt.savefig('cma_es_pic/0'+str(gen)+'.png')
        elif(len(str(gen))) == 3:
            plt.savefig('cma_es_pic/'+str(gen)+'.png')
        #savefig('cma_es_pic/figure'+str(gen)+'.png')
        plt.clf()
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
