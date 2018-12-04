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
        self.ax.scatter(xy[0],xy[1],c='red',marker='.')

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
    min_fit = min(fitness)
    nvmap = noveltymap(5,POPNUM)

    print("gen ","min ""max ","mean")
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
            _pop = nvmap.popInd(pop)
            for off ,_ind in zip(offspring,_pop):
                off = _ind.xy
                off.fitness.values = _ind.novelty,
                pop[:] = offspring
                fits = [benchmarks.himmelblau(ind)[0] for ind in pop]
        else:
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]

            for ind in invalid_ind:
                nvmap.ax.scatter(ind[0],ind[1],c='blue',marker='.')
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
        if (min(fits) - min_fit) == 0:
            novflag += 1
        else :
            novflag = 0
        print(i ,min(fits) ,max(fits) ,mean)
        min_fit = min(fits)
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
2回連続で最小値が一致していたらnovelty_searchを行う．

0 4.075023050190184 1156.1453120427354 166.4136630695671
1 0.022914707811938422 659.4723034874191 87.58800368033285
2 0.022914707811938422 565.286954344603 46.92552132727429
3 0.022914707811938422 1155.8698271088651 32.86141289300736
4 0.16929126310766854 130.72871631851797 10.603308629914354
5 0.26926918751105927 116.1383051077541 13.440188120269836
6 0.0790438329543217 226.68121785607093 11.451844704279138
7 0.04616787809424775 126.51163292258249 8.864807308145636
8 0.042111253129909064 69.50050986331198 3.726453945503491
9 0.005903559769173528 86.89761470235598 1.2720889780771634
10 0.005863780260742529 0.6259415090656157 0.12919107959593795
11 0.0034695191889880473 0.2950602606908829 0.056390570944695864
12 0.0005776901895934824 0.3547434116543602 0.030428990674512525
13 0.0005378153042933462 0.06305066581447914 0.011593879716016227
14 2.096055239188248e-05 0.019339826801054928 0.004678405100041424
15 2.096055239188248e-05 0.8786182447744044 0.0086581505029607
16 2.096055239188248e-05 0.007619639099540984 0.001388039768786345
17 2.096055239188248e-05 0.002774892726371434 0.0006048221818924921
18 2.558009484096637e-06 0.0061048135369942075 0.0005366203134837156
19 2.558009484096637e-06 0.0009627825903106884 0.0001646154093970186
20 6.583422362609945e-07 0.06218729843859929 0.0004652484826122095
21 6.583422362609945e-07 9.031716046428972e-05 1.9993480287989535e-05
22 1.3800267137828816e-07 5.209852276812126e-05 9.48709349123559e-06
23 3.3897870493270764e-08 3.432699701712737e-05 3.070710682560426e-06
24 4.2486774948657064e-08 5.15698609901464e-06 1.185928585670137e-06
25 1.2909539516513869e-09 2.7809857554533547e-06 5.454746402156326e-07
26 1.2909539516513869e-09 6.564882129754862 0.04376611121007812
27 1.2909539516513869e-09 6.37049221514567e-07 9.387104412834435e-08
28 8.024038720681296e-10 0.11033640444617202 0.000735617583952503
29 3.315620891301026e-10 2.606648049395684 0.017377690078249325
30 2.88670090235261e-10 1.4137442739746308e-07 1.4948849959025667e-08
31 5.94102170041253e-11 3.2375781537962084e-08 4.9855474093788e-09
32 1.6701714787758792e-11 16.934000285097905 0.1128933365611537
33 1.3968885086446837e-11 4.4030690633126624e-09 4.769435642673693e-10
34 5.206903612055012e-12 3.1117000394018877e-09 1.97864209619763e-10
35 1.6796971705353068e-12 4.0571958805598644e-10 7.426209213061056e-11
36 1.6796971705353068e-12 1.842365760369841e-10 3.338080297059341e-11
37 4.719236057535443e-13 7.314570244847485e-11 1.3052356115497193e-11
38 2.459635763556445e-13 4.9851236628093714e-11 6.5356333829396645e-12
39 1.6787771184742675e-13 1.4409699912341533e-11 2.5261225482944097e-12
40 1.999319172838828e-14 1.1669216363914504e-11 1.487409050919017e-12
41 4.489566701514236e-15 9.503382168077711e-12 7.051820634966943e-13
42 2.4501715366570282e-15 2.3955085285643153e-12 3.330600279996327e-13
43 1.8738519156068178e-16 7.921659636636069e-13 1.0875903805847064e-13
44 1.8738519156068178e-16 3.2397173703012837e-13 4.3365114298911424e-14
45 1.8738519156068178e-16 1.4482582496534703e-13 1.5830280986668614e-14
46 1.8738519156068178e-16 3.0222771181959523e-14 4.8970202396572455e-15
47 3.6713616518984705e-17 3.0222771181959523e-14 3.821896579829602e-15
48 2.496355579875368e-17 5.9148287936370244e-15 1.6327379593773215e-15
49 1.4379353917959893e-17 4.838582841308113e-15 9.508659113321933e-16
50 1.4379353917959893e-17 1.7308747902123215e-15 3.9968017157563733e-16
51 6.534982919625112e-18 0.10619738790124479 0.0007079825860084797
52 1.488662771958017e-19 3.3550191233153094e-16 6.60046265912534e-17
53 1.488662771958017e-19 2.35978480455567e-16 2.6137961024745244e-17
54 1.488662771958017e-19 8.804912457824704e-17 1.0998618820920322e-17
55 2.2812792811200066e-22 5.416057772123956e-17 5.109836009762378e-18
56 2.407739117111989e-21 0.013031852619248423 8.687901746165977e-05
57 2.407739117111989e-21 8.59993513457071 0.05733290089713807
58 5.2815659486885356e-21 8.35200469001697e-18 3.822746609279668e-19
59 2.844083132490054e-22 5.872237546300047e-19 8.637920509864395e-20
60 9.680820959490377e-23 2.1014465402378133e-19 3.210003900645322e-20
61 9.680820959490377e-23 7.693109642944362e-20 1.454376539319853e-20
62 2.4144949716025388e-24 3.016968255220673e-20 4.8913401484212755e-21
63 8.713145403008845e-24 1.0536386367210329 0.007024257578140219
64 7.604243628539669e-24 2.9131449875345262e-21 4.388778688805641e-22
65 1.6520459183492317e-24 1.0138993850265983e-21 1.8051997722832042e-22
66 2.9358562993067264e-25 6.45768747202594e-22 6.105574689592328e-23
67 7.486289990547402e-26 1.3169345241499801e-22 1.936798490760229e-23
68 7.486289990547402e-26 3.216985499999597e-23 7.40405127799162e-24
69 2.4038169503894672e-26 2.2697662111957132e-23 3.70038841866636e-24
70 2.4038169503894672e-26 1.0673821317612219e-23 1.8460820615018074e-24
71 9.486052385282667e-27 1.0009078328822083 0.0066727188858813884
72 9.150786500563737e-27 1.6995724267628225 0.011330482845085484
73 4.291403324402304e-27 0.34293190732222867 0.0022862127154815244
74 1.672385119068545e-28 2.0781751687142335e-25 2.92044984582776e-26
75 1.672385119068545e-28 4.2828836266259173e-26 1.1797057220636532e-26
76 7.020862056467005e-29 2.7657463337048674e-26 6.176181353592554e-27
77 1.5777218104420236e-29 2.133079887717616e-26 2.4520742014981873e-27
78 3.1554436208840472e-30 6.147593034387345e-27 5.83615074900609e-28
79 3.1554436208840472e-30 7.131302583197947e-28 1.4240517061049706e-28
80 7.888609052210118e-31 3.155443620884047e-28 5.65718450497495e-29
81 0.0 1.672385119068545e-28 2.7394509701975003e-29
"""
