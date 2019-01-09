import operator
import random

import numpy

from deap import base
from deap import benchmarks
from deap import creator
from deap import tools

import matplotlib.pyplot as plt

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Particle", list, fitness=creator.FitnessMin, speed=list,
    smin=None, smax=None, best=None)

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
toolbox.register("particle", generate, size=2, pmin=-6, pmax=6, smin=-3, smax=3)
toolbox.register("population", tools.initRepeat, list, toolbox.particle)
toolbox.register("update", updateParticle, phi1=2.0, phi2=2.0)
toolbox.register("evaluate", benchmarks.himmelblau)

def main():
    pop = toolbox.population(n=150)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)

    logbook = tools.Logbook()
    logbook.header = ["gen", "evals"] + stats.fields

    GEN = 100
    best = None

    x = numpy.arange(-6, 6, 0.05) #x軸の描画範囲の生成。0から10まで0.05刻み。
    y = numpy.arange(-6, 6, 0.05) #y軸の描画範囲の生成。0から10まで0.05刻み。
    sum_min = 0
    X, Y = numpy.meshgrid(x, y)
    for gen in range(GEN):
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
        plt.pcolormesh(X, Y, Z, cmap='hsv') # 等高線図の生成。cmapで色付けの規則を指定する。
        #plt.pcolormesh(X, Y, Z,cmap='hsv') # 等高線図の生成。cmapで色付けの規則を指定す
        #pp=plt.colorbar (orientation="vertical") # カラーバーの表示
        #pp.set_label("Label", fontname="Arial", fontsize=24) #カラーバーのラベル
        """
        plt.xlabel('X', fontsize=24)
        plt.ylabel('Y', fontsize=24)
        """
        for part in pop:
            part.fitness.values = toolbox.evaluate(part)
            if not part.best or part.best.fitness < part.fitness:
                part.best = creator.Particle(part)
                part.best.fitness.values = part.fitness.values
            if not best or best.fitness < part.fitness:
                best = creator.Particle(part)
                best.fitness.values = part.fitness.values
        for part in pop:
            toolbox.update(part, best)

        for ind in pop:
            ax.scatter(ind[0],ind[1],c='blue',marker='.')

        # Gather all the fitnesses in one list and print the stats
        logbook.record(gen=gen, evals=len(pop), **stats.compile(pop))
        sum_min += (logbook.select("min")[0])

        if len(str(gen))==1:
            plt.savefig('pso_pic/00'+str(gen)+'.png')
        elif(len(str(gen))) == 2:
            plt.savefig('pso_pic/0'+str(gen)+'.png')
        elif(len(str(gen))) == 3:
            plt.savefig('pso_pic/'+str(gen)+'.png')

        #savefig('cma_es_pic/figure'+str(gen)+'.png')
        plt.clf()
        plt.close()
    print(sum_min / 100)
    return pop, logbook, best

if __name__ == "__main__":
    main()

"""
gen     evals   avg     std     min     max
0       100     322.207 336.322 6.36835 1679.1
1       100     253.476 470.145 1.98338 2732.76
2       100     152.911 151.08  5.25208 1122.92
3       100     142.409 217.597 1.98338 1264.56
4       100     402.083 452.363 1.54254 2272.52
5       100     254.938 434.029 1.89224 2907.06
6       100     199.753 301.642 2.62743 1774.64
7       100     225.651 330.382 2.19357 2095.27
8       100     248.52  351.023 3.25223 1594.69
9       100     211.738 275.081 0.589135        1287
10      100     184.791 333.849 0.273232        1978.27
11      100     192.116 289.615 6.90301         1696.93
12      100     244.262 333.081 0.831126        1838.06
13      100     206.991 249.364 1.09653         1114
14      100     150.216 182.436 0.0532021       1212.72
15      100     147.77  238.637 4.98591         1783.86
16      100     145.071 184.701 6.47554         974.42
17      100     203.902 298.545 1.42032         1814.77
18      100     132.262 176.193 0.811128        1165.31
19      100     177.954 261.105 0.731348        1541.14
20      100     175.074 214.27  0.35504         828.44
21      100     148.818 182.178 0.605923        1012.25
22      100     145.596 206.326 0.851168        1185.04
23      100     142.674 202.765 0.00896042      1301.62
24      100     212.642 307.346 3.24097         1982.93
25      100     131.396 178.132 0.185907        959.858
26      100     211.335 421.823 0.0691946       3039.65
27      100     158.502 225.159 0.570058        1161.64
28      100     206.338 313.896 1.75257         1902.57
29      100     208.856 338.384 0.504579        2253.99
30      100     159.199 224.249 0.392212        1020.65
31      100     191.324 364.363 1.16942         3084.28
32      100     198.56  363.013 0.321867        2240.23
33      100     176.874 241.009 0.991258        1275.72
34      100     149.758 276.692 0.0967464       2400.27
35      100     194.751 297.838 0.0193054       2009.95
36      100     161.96  218.702 0.0627807       1472.46
37      100     186.928 264.211 0.0233926       1380.36
38      100     122.842 154.31  0.407105        965.926
39      100     136.328 160.349 0.0372514       759.412
40      100     191.031 271.704 1.13872         1454.9
41      100     171.563 198.644 2.03953         942.02
42      100     192.626 330.491 4.57185         2710.61
43      100     189.398 338.909 3.72369         2581.19
44      100     198.748 384.847 2.54107         3260.06
45      100     155.614 215.561 0.410384        1114.5
46      100     188.854 351.935 4.2095          2885.3
47      100     153.579 168.943 0.235192        896.696
48      100     225.773 391.818 2.25624         2557.55
49      100     181.012 238.693 0.665635        1558.81
50      100     173.166 223.573 0.706008        906.583
51      100     134.603 226.465 0.459941        1469.23
52      100     190.122 284.107 1.40953         2316.97
53      100     171.064 224.414 0.164306        1256.3
54      100     212.424 304.685 0.310922        1781.88
55      100     192.995 354.806 0.335896        2224.54
56      100     156.754 199.63  1.15298         921.038
57      100     172.638 257.382 0.482223        1525.67
58      100     199.523 378.555 0.391947        3435.56
59      100     157.272 218.672 0.0368538       1434.71
60      100     164.843 206.822 1.76546         1182.2
61      100     160.782 218.989 0.568301        1073.19
62      100     214.678 383.515 0.0637431       2463.03
63      100     141.312 175.909 0.0460617       954.699
64      100     201.466 330.109 0.045026        2338.2
65      100     203.591 297.255 0.437969        1286.03
66      100     167.512 210.36  1.53879         821.099
67      100     158.129 232.722 0.404479        1770.19
68      100     198.24  262.084 1.2627          1453.93
69      100     169.803 253.822 0.493308        1583.81
70      100     163.302 230.931 0.0951354       1445.29
71      100     163.473 305.93  0.227401        2055.42
72      100     243.824 330.286 0.00178629      1903.97
73      100     202.748 334.457 0.0966541       1877.26
74      100     162.179 272.877 0.0705197       1579.57
75      100     140.087 202.045 0.220018        1084.3
76      100     162.508 186.546 0.00182453      1127.99
77      100     227.044 411.42  0.236832        3129.77
78      100     233.771 370.542 0.00169898      2397.7
79      100     157.939 246.909 0.226126        1196.5
80      100     161.354 310.202 0.00192072      2548.89
81      100     229.748 343.85  4.54949         2160.24
82      100     182.481 331.405 0.00206431      2478.49
83      100     163.059 247.083 0.729141        1514.36
84      100     145.258 251.46  2.52237         1878.17
85      100     164.908 195.999 0.278011        944.207
86      100     173.285 271.166 1.09505         1821.11
87      100     150.657 206.458 0.323687        1454.77
88      100     163.762 356.101 0.313392        3155.39
89      100     213.979 341.812 2.14068         2211.82
90      100     218.573 491.494 1.09751         4327.76
91      100     200.949 329.983 0.227288        2537.79
92      100     164.757 243.391 0.070475        1201.77
93      100     195.961 272.808 0.252811        1535.29
94      100     230.017 412.949 1.5195          3380.03
95      100     245.195 474.019 4.17959         3764.8
96      100     207.346 481.226 0.320075        3425.14
97      100     168.391 273.384 6.47205         2208.28
98      100     150.058 198.723 0.121146        1054.65
99      100     163.463 249.175 1.15747         1504.2
"""
