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
loop = 8
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

    print("gen ","min ""max ","mean")
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
        if i % loop == 0:
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
n回に1回novelty_searchを行い，それ以外はGA今回は8回

0 1.640462758817756 4513.65597170299 203.60591260542202
1 1.640462758817756 1472.9208813547261 111.70329440038375
2 0.02185413254050248 793.2010088997904 66.12294273185825
3 0.02185413254050248 767.6493716524137 49.714454616948586
4 0.02185413254050248 877.7180336090082 21.09576666606099
5 0.024899186753266675 1128.4318813519276 42.648835407799645
6 0.024899186753266675 671.6699964069588 15.437125421402385
7 0.024899186753266675 952.2665100094607 21.829823708331844
8 0.0029766692281397505 545.5094530959386 8.89494525523739
9 0.0016614438841939666 545.5094530959386 13.721432017370935
10 0.0016614438841939666 780.6435783781858 20.990664703556604
11 0.001194198777423114 0.28410125812594306 0.029714413709549133
12 0.0007733240196398175 0.07502522938568867 0.019155591784257346
13 0.00015068888472081967 0.04933156305527415 0.011794648374384483
14 0.00015068888472081967 0.05654960198331968 0.005889583465951981
15 2.930987299745263e-05 6.166486884072961 0.04315253526926105
16 2.092133704768315e-05 0.004795553780449664 0.0010173084688034453
17 1.3117326198397397e-06 0.0023857582443886985 0.0006613298160529312
18 6.846488714971138e-07 0.0026740969265992374 0.0002957670434736453
19 1.3117326198397397e-06 0.0007127171741996276 8.366396839080263e-05
20 2.785431528938294e-07 0.0003037570628189306 3.5075790925393666e-05
21 3.107810092924678e-07 0.00015410687182760953 1.3437551784203615e-05
22 1.1059758182080113e-08 3.613221935913704e-05 4.684653658518768e-06
23 4.309801778016188e-09 9.158504915156126e-06 1.5218276245548663e-06
24 4.309801778016188e-09 8.322352222594557 0.055482857726049474
25 7.402584893124437e-10 3.132961260494178e-06 5.116595132288646e-07
26 1.6400913630216597e-09 1.642554551905212e-06 1.687085288972746e-07
27 1.985531020404132e-10 4.832577582625344e-07 3.801137137858463e-08
28 7.927515229032327e-11 6.024043832125199e-08 7.991621933939957e-09
29 2.7019155040321796e-11 1.2884747523174026e-08 2.3403796348521955e-09
30 1.8378845211009148e-11 4.836927743726167e-09 8.842056729139098e-10
31 1.1117337813777594e-11 5.4552647396489e-09 3.394388872328857e-10
32 2.4499915288805414e-13 7.310844461962506e-10 1.252216807634119e-10
33 4.344902797959445e-12 8.0583881124809e-10 1.0685794582609126e-10
34 4.785234588231514e-13 2.3882834450753214e-10 3.874120482166856e-11
35 4.785234588231514e-13 7.094663751051623e-11 1.5628187143909893e-11
36 1.0040499664247485e-14 7.378987633564401e-11 6.821083620912452e-12
37 7.372809409344575e-15 1.5351727088810617e-11 2.1003186875707136e-12
38 9.379696286290359e-15 4.5794060220354815e-12 5.664227258198185e-13
39 1.0867458271922789e-15 2.4173895855430205e-12 2.608786099660133e-13
40 5.724522142137901e-17 7.99996337114439e-13 1.090834011531066e-13
41 1.0867458271922789e-15 3.3789992819594463e-13 9.08648888195572e-14
42 4.321655738811997e-16 2.403436257189969e-13 3.8068659769844325e-14
43 1.1434168224951644e-16 1.0399194735824924e-13 1.122462832522266e-14
44 3.99000443886333e-17 2.7556088152636708e-14 3.2679542545937964e-15
45 2.8926984753037636e-17 12.297478696420225 0.08198319130946899
46 1.1480611998205015e-17 2.8093436374925497e-15 3.226935904801536e-16
47 6.527837590438687e-18 7.582495786291058e-16 1.2393703082141783e-16
48 5.79983078807071e-19 3.595949101602152e-16 6.293817933942e-17
49 2.7130953896490025e-18 0.8621789137213602 0.0057478594248091125
50 6.240096449586208e-19 8.461736964623204e-17 1.932185306908045e-17
51 5.3376040968874245e-20 2.7165959417798508e-17 8.232274638741175e-18
52 2.4486172803776952e-20 3.2756703703828923e-17 4.656791306380741e-18
53 2.4486172803776952e-20 0.2628146996297556 0.0017520979975317046
54 9.344271645410307e-21 4.77598078748985e-18 8.135827316794889e-19
55 2.3304944907280544e-21 3.6323364126581836e-18 3.8178765029555055e-19
56 1.5198965674398883e-21 1.5664242679413505 0.010442828452942336
57 8.99937683770755e-22 6.291794883861553e-19 7.050421295433894e-20
58 2.8748376952954367e-22 9.134006509616754 0.060893376730778354
59 2.8748376952954367e-22 1.2992161263124562 0.008661440842083042
60 6.474834625978367e-24 1.7264888028341283e-20 3.791043821113988e-21
61 7.9589438905196235e-25 9.344271645410307e-21 1.6966941819557884e-21
62 3.099537048531034e-25 3.759866285033027e-21 7.8264917897832135e-22
63 1.741836433164203e-25 1.723698546434326e-21 2.917479941502923e-22
64 4.6034766985077365e-26 1.3192388752065521e-21 1.0282341408313353e-22
65 3.666941031829351e-26 6.598690111055826e-22 4.3741592367815613e-23
66 3.666941031829351e-26 5.836441365363573e-23 7.54023047960726e-24
67 9.557838727657779e-27 2.184649933902448e-23 1.776328240285288e-24
68 1.2621774483536189e-27 4.0826707288808245e-24 4.809329899134432e-25
69 5.3090338921374095e-28 9.57579320186061e-25 1.242878334442476e-25
70 9.150786500563737e-29 2.994515996218961e-25 4.600872931613237e-26
71 2.8398992587956425e-29 0.7729035358475103 0.005152690238983401
72 1.262177448353619e-29 3.666941031829351e-26 5.143546651440146e-27
73 2.8398992587956425e-29 3.666941031829351e-26 4.412272592300268e-27
74 0.0 0.5780755688068664 0.003853837125379109
"""
