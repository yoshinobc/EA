import numpy as np

from deap import algorithms
from deap import base
from deap import benchmarks
from deap import cma
from deap import creator
from deap import tools
import time
import matplotlib.pyplot as plt

N = 7  # 問題の次元
NGEN = 200  # 総ステップ数

def func(pop):
      return (1 - 1 / (1 * np.linalg.norm(np.array(pop) - [-2,-2,-2,-2,-2,-2,-2], axis=0) + 1)) + (1 - 1 / (2 *np.linalg.norm(np.array(pop) - [4,4,4,4,4,4,4],  axis=0) + 1)),

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("evaluate", benchmarks.himmelblau)

def main():
    np.random.seed(64)

    # The CMA-ES algorithm
    strategy = cma.Strategy(centroid=[5.0]*N, sigma=3.0, lambda_=20*N)
    toolbox.register("generate", strategy.generate, creator.Individual)
    toolbox.register("update", strategy.update)

    halloffame = tools.HallOfFame(1)

    halloffame_array = []
    C_array = []
    centroid_array = []
    ok_count = 0
    stop_gen = 200
    #print("gen ","min ","max ","mean ","std")
    for gen in range(NGEN):

        # 新たな世代の個体群を生成
        population = toolbox.generate()
        # 個体群の評価

        fitnesses = toolbox.map(func, population)
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit

        fits = [ind.fitness.values[0] for ind in population]
        length = len(population)
        mean = sum(fits) / length
        sum2 = sum(x*x for x in fits)
        std = abs(sum2 / length - mean**2)**0.5
        #print(gen ,min(fits) ,max(fits) ,mean,std)

        if min(fits) <= np.exp(-10) :
            ok_count = 1
            stop_gen = gen
            break

        # 個体群の評価から次世代の計算のためのパラメタ更新
        toolbox.update(population)

        # hall-of-fameの更新
        halloffame.update(population)

        halloffame_array.append(halloffame[0])
        C_array.append(strategy.C)
        centroid_array.append(strategy.centroid)
    return population,ok_count,stop_gen
    # 計算結果を描画

if __name__ == "__main__":
    count = 0
    sum_gen = 0
    start = time.time()
    for i in range(1000):
        print(i)
        pop,ok_count,stop_gen = main()
        count += ok_count
        sum_gen += stop_gen
    episode_time = time.time() - start
    print("count",count)
    print("sum_gen",sum_gen / 1000)
    print("time",episode_time)
