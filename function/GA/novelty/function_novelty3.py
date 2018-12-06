import numpy as np
from deap import algorithms,base,cma,creator,tools,benchmarks
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
import time

def _update_plot(i,fig):
    plt.scatter(xy[0],xy[1],c="red",marker='.')

class novind:
    def __init__(self,xy,novelty,fitness):
        self.xy = xy
        self.novelty = novelty
        self.fitness = fitness
        self.score = 0

    def inscore(self,score):
        self.score = score

    def __repr__(self):
        return repr((self.xy,self.novelty,self.fitness,self.score))

class noveltymap:
    def __init__(self,k,popnum):
        print("make noveltymap")
        self.map = []
        self.k = k
        self.popnum = popnum
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(1,1,1)

    def distance(self,p0,p1):
        return np.sqrt((p0[0] - p1[0])**2 + (p0[1] - p1[1]) **2)

    def insertInd(self,xy,novelty,fitness):
        self.map.append(novind(xy,novelty,fitness))
        #ani = animation.FuncAnimation(self.fig,_update_plot,fargs = (self.fig),frames = 360,interval=1)
        plt.scatter(xy[0],xy[1],c="red",marker='.')
    def max_min_nov(self):
          return sorted(self.map,key=lambda u:u.novelty)[0].novelty,sorted(self.map,key=lambda u:u.novelty)[len(self.map) - 1].novelty

    def max_min_fit(self):
          return sorted(self.map,key=lambda u:u.fitness)[0].fitness,sorted(self.map,key=lambda u:u.fitness)[len(self.map) - 1].fitness

    def calFit(self,ind):
        return benchmarks.himmelblau(ind)[0]

    def popInd(self,population):
        for ind in population:
            distances = np.array([self.distance(p.xy,ind) for p in self.map])
            nearest_distance = sorted(distances)[:self.k]
            novelty = sum(nearest_distance) / self.k
            fitness = self.calFit(ind)
            self.insertInd(ind,novelty,fitness)
            #self.insertInd(ind,novelty)
            min_nov, max_nov = self.max_min_nov()
            min_fit, max_fit = self.max_min_fit()
            score = (1 - ro) * ((fitness - min_fit) / (max_fit - min_fit + 0.0001)) + ro * ((novelty - min_nov) / (max_nov - min_nov + 0.0001))
            novind.inscore(self,score)
        return sorted(self.map,key=lambda u:u.score)[:self.popnum]


NGEN = 100
POPNUM = 150
N = 2
INDSIZE = N
CXPB = 0.5
MUTPB = 0.01
ro = 0.2


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

    nvmap = noveltymap(3,POPNUM)

    print("gen ","min ","max ","mean")
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
        if i % 1 == 0:
            _pop = nvmap.popInd(pop)
            for off ,_ind in zip(offspring,_pop):
                off = _ind.xy
                off.fitness.values = _ind.novelty,
                pop[:] = offspring
                fits = [benchmarks.himmelblau(ind)[0] for ind in pop]
        else:
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]

            fitness = list(map(toolbox.evaluate,np.clip(invalid_ind,-6,6)))
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
        print(i ,min(fits) ,max(fits) ,mean)
        if min(fits) == 0:
            break
        #print("gen:",i,"  Min %s" % min(fits),"  Max %s" % max(fits),"  Avg %s" % mean)
        #print("gen:",i,"  Min %s" % min(fits),"  Max %s" % max(fits),"  Avg %s" % mean,"  Std %s" % std)
        #print(i,max(fits),mean)
    nvmap.fig.show()
    time.sleep(100000000)
    return pop,hof

if __name__=='__main__':
    print("pop_num = ",POPNUM)
    print("gen_num ",NGEN)
    pop,hof = main()
    expr = tools.selBest(pop,1)[0]
    print(expr)

"""
fitnessとnoveltyを織り交ぜたものを使った
0 0.7695233205263567 4226.435236047047 278.4876458708801
1 0.7695233205263567 4226.435236047047 261.06822965323045
2 0.7695233205263567 2836.120261980366 154.67090294193383
3 0.7695233205263567 2505.9345131368314 185.89519917387074
4 0.7695233205263567 4472.903865064469 180.7149406855123
5 0.13726256626504402 5727.630142476004 176.53159875282634
6 0.7695233205263567 832.3960750696077 45.2701912486131
7 0.41416649579829956 2631.5136772655796 69.2538949293882
8 0.14590643246000173 2592.447297602403 54.80409940928164
9 0.05661714879292399 1025.1350161214245 23.17461117796989
10 0.4542912606591756 1025.1350161214245 17.095944020658166
11 0.31696208225363365 160.6873744082568 9.705976441826456
12 0.05784671339373896 1493.6163416822387 17.08795698316863
13 0.05812209872101766 126.27396029990734 3.371999507910691
14 0.05812209872101766 126.27396029990734 3.076272216590186
15 0.05812209872101766 331.88902843656075 4.469090797117224
16 0.4880863871107839 331.88902843656075 4.752044501506297
17 0.06181036999798062 22.07684115575269 1.629074396136358
18 0.6151456073884547 20.37261544974412 1.1172547653255351
19 0.24892302612515066 13.63609575236174 0.8807562171600745
20 0.29730551532455485 0.9655349897455219 0.7696827844855776
21 0.5573474179181327 0.8738609037690195 0.7662673287617198
22 0.7603744330265035 0.8428042722694884 0.7703385499765527
23 0.7695233205263508 0.7745799810744775 0.7695570315966783
24 0.7695233205263508 0.7695233205263599 0.7695233205263573
25 0.7695233205263508 0.7695233205263662 0.7695233205263573
26 0.7695233205263473 0.7695233205263599 0.7695233205263573
27 0.0048896909104536234 0.7695233205263599 0.7644257629955847
28 0.7695233205263567 0.7695233205263662 0.7695233205263576
29 0.7695233205263508 1.2730502438249465 0.7728801666816814
30 0.7695233205263567 0.7695233205263599 0.7695233205263575
31 0.7695233205263508 0.7695233205263599 0.7695233205263571
32 0.7695233205263508 0.7695233205263599 0.7695233205263573
33 0.7695233205263473 0.7695233205263567 0.7695233205263573
34 0.7695233205263508 0.7695233205263599 0.7695233205263575
35 0.7695233205263508 0.7695233205263662 0.7695233205263576
36 0.7695233205263508 0.7695233205263599 0.7695233205263574
37 0.7695233205263508 0.7695233205263599 0.7695233205263574
38 0.7695233205263508 0.7695233205263599 0.7695233205263575
39 0.7695233205263508 0.7695233205263599 0.7695233205263576
40 0.7695233205263508 0.7695233205263599 0.7695233205263572
41 0.7695233205263473 0.7695233205263599 0.7695233205263575
42 0.7695233205263473 0.7695233205263599 0.7695233205263576
43 0.7695233205263508 0.7695233205263599 0.7695233205263573
44 0.7695233205263508 0.7695233205263599 0.7695233205263575
45 0.7695233205263508 0.7695233205263599 0.7695233205263574
46 0.7695233205263508 0.7695233205263599 0.7695233205263575
47 0.7695233205263508 3.115625292213908 0.7851640003376078
48 0.7695233205263508 0.7695233205263599 0.7695233205263574
49 0.7695233205263567 0.7695233205263599 0.7695233205263576
50 0.7695233205263508 0.7695233205263599 0.7695233205263575
51 0.7695233205263473 10.560378296044812 0.8347956870298145
52 0.7695233205263473 0.7695233205263599 0.7695233205263574
53 0.7695233205263567 0.7695233205263599 0.7695233205263576
54 0.7695233205263508 0.7695233205263599 0.7695233205263575
55 0.7695233205263508 0.7695233205263599 0.7695233205263573
56 0.7695233205263508 0.7695233205263599 0.7695233205263576
57 0.7695233205263473 0.7695233205263599 0.7695233205263573
58 0.7695233205263508 0.7695233205263567 0.7695233205263574
59 0.7695233205263508 0.7695233205263599 0.7695233205263574
60 0.7695233205263567 0.7695233205263599 0.7695233205263576
61 0.7695233205263508 0.7695233205263567 0.7695233205263573
62 0.7695233205263508 0.7695233205263599 0.7695233205263575
63 0.7695233205263473 0.7695233205263599 0.7695233205263575
64 0.7695233205263473 0.7695233205263599 0.7695233205263572
65 0.7695233205263508 0.7695233205263662 0.7695233205263567
66 0.7695233205263508 0.7695233205263567 0.7695233205263571
67 0.7695233205263508 0.7695233205263662 0.7695233205263574
68 0.7695233205263508 0.7695233205263662 0.7695233205263573
69 0.7695233205263567 0.7695233205263662 0.7695233205263576
70 0.7695233205263508 1.164253809478277 0.7721548571193702
71 0.6728506956726139 1.164253809478277 0.7730078968540992
72 0.6728506956726139 0.7695233205263599 0.7688788363606658
73 0.7695233205263508 0.7695233205263599 0.7695233205263575
74 0.7695233205263508 0.7695233205263599 0.7695233205263575
75 0.7695233205263473 0.7695233205263599 0.7695233205263574
76 0.7695233205263508 0.7695233205263599 0.7695233205263576
77 0.7695233205263508 0.7695233205263567 0.7695233205263571
78 0.7695233205263508 0.7695233205263599 0.7695233205263575
79 0.7695233205263567 19.618113416625686 0.8951805878336865
80 0.28112496843132573 0.7695233205263599 0.7662673315123907
81 0.7695233205263567 0.7695233205263599 0.7695233205263576
82 0.7695233205263508 0.7695233205263599 0.7695233205263574
83 0.7695233205263508 0.7695233205263567 0.7695233205263573
84 0.7695233205263508 0.7695233205263599 0.7695233205263575
85 0.7695233205263508 0.7695233205263599 0.7695233205263575
86 0.7695233205263508 0.7695233205263599 0.7695233205263571
87 0.7695233205263508 0.7695233205263599 0.7695233205263574
88 0.7695233205263508 0.7695233205263599 0.7695233205263575
89 0.7695233205263508 0.7695233205263599 0.7695233205263576
90 0.7695233205263508 2.058712103361783 0.778117912411927
91 0.7695233205263473 0.7695233205263662 0.769523320526357
92 0.7695233205263508 0.7695233205263662 0.7695233205263576
93 0.7695233205263508 0.7695233205263599 0.7695233205263573
94 0.7695233205263508 0.7695233205263599 0.7695233205263576
95 0.7695233205263567 0.7695233205263599 0.7695233205263575
96 0.7695233205263508 0.7695233205263599 0.7695233205263576
97 0.7695233205263508 0.7695233205263599 0.7695233205263573
98 0.7695233205263508 0.7695233205263599 0.7695233205263573
99 0.7695233205263508 0.7695233205263599 0.7695233205263574
"""
