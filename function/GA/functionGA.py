import numpy as np
from deap import algorithms,base,cma,creator,tools,benchmarks
import matplotlib.pyplot as plt
import random

NGEN = 100
POPNUM = 15
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
    #pop,hof = algorithms.eaSimple(pop,toolbox,cxpb=0.5,mutpb=0.01,ngen=200,stats=stats,halloffame=hof,verbose=True)
    fitness = list(map(toolbox.evaluate,np.clip(pop,-6,6)))

    for ind, fit in zip(pop, fitness):
        ind.fitness.values = fit

    print("gen ","min ","max ","mean")
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    for ind in pop:
        ax.scatter(ind[0],ind[1],c='blue',marker='.')
    for i in range(NGEN):
        # Select the next generation individuals
        print(pop)
        offspring = toolbox.select(pop, len(pop))
        print(offspring)
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
        for ind in invalid_ind:
            ax.scatter(ind[0],ind[1],c='blue',marker='.')
        """
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
        sum2 = sum(x*x for x in fits)
        std = abs(sum2 / length - mean**2)**0.5
        print(i ,min(fits) ,max(fits) ,mean)
        if min(fits) == 0:
            break
        #print("gen:",i,"  Min %s" % min(fits),"  Max %s" % max(fits),"  Avg %s" % mean,"  Std %s" % std)
        #print(i,max(fits),mean)
    plt.show()
    return pop,hof

if __name__=='__main__':
    print("pop_num = ",POPNUM)
    print("gen_num ",NGEN)
    pop,hof = main()
    expr = tools.selBest(pop,1)[0]
    print(expr)
"""
0 0.7086831326761399 1039.229710314044 138.6544885556421
1 0.7086831326761399 919.7960013500491 94.41784448712171
2 0.6885714570806267 668.2286325927214 51.36592146834823
3 0.6885714570806267 723.5177133507343 50.86906092813196
4 0.025185965739695518 779.4107167072684 29.228246087124298
5 0.025185965739695518 741.2371195705961 27.659307693594982
6 0.01587120481328802 96.07647314011285 3.1565760373160425
7 0.01587120481328802 6.248244891470439 0.6169488287034409
8 0.003845091905112688 1.2806383797242433 0.27486456949461685
9 7.992811017322151e-05 9.362127702257428 0.16505966369521138
10 7.992811017322151e-05 0.3582160590333061 0.03894335925536399
11 6.675251026510311e-05 0.08537928470987959 0.01580922259436695
12 4.966407852285759e-05 0.04808609106188921 0.007303539428251039
13 1.7164830085630763e-05 10.000877940537588 0.06918806761797754
14 1.7164830085630763e-05 0.005563876847074348 0.0010887482334498587
15 1.3269332532896833e-05 0.0033113931174862934 0.0004567357525850945
16 3.363865657911146e-06 0.0010205389913248257 0.0001505483136858111
17 6.632875437595263e-08 0.000340628366730063 4.8903463236970954e-05
18 6.632875437595263e-08 0.00016232998428392518 2.4415184421536547e-05
19 6.632875437595263e-08 7.631293579057109e-05 1.2633327639702984e-05
20 6.408073828461385e-08 1.900653788337578e-05 5.360938402932036e-06
21 9.08889887176791e-09 1.1220161416457357e-05 2.897358257863171e-06
22 3.642925270867583e-08 5.367482049823598e-06 1.7765561409098934e-06
23 3.642925270867583e-08 7.4561122312896044e-06 9.695260970284602e-07
24 3.025923321612684e-09 2.899636198463108e-06 5.416885168129678e-07
25 2.421227914627268e-09 1.1412253876716788e-06 2.0976235529318816e-07
26 1.8767097609551756e-09 4.395926249397808e-07 7.265999847236268e-08
27 1.8767097609551756e-09 1.6359301172903698e-07 3.307026991661323e-08
28 3.293128182773096e-10 7.08942561966769e-08 1.775373439776159e-08
29 1.9635671861244792e-10 6.01469920394756e-08 8.112985889955457e-09
30 2.039780533774342e-12 2.332415679604431e-08 3.125134601762383e-09
31 1.712571758131481e-11 6.963713673671091e-09 1.0943386360046164e-09
32 7.146606871494256e-13 3.4849163417433433e-09 4.283755610901973e-10
33 7.352662972172639e-14 2.8106648455922947e-09 1.960182037846878e-10
34 7.352662972172639e-14 4.507182594177267e-10 5.418355204596572e-11
35 7.352662972172639e-14 1.570548512450084e-10 1.7122787251351664e-11
36 4.025379899310818e-14 2.90001179886498e-11 5.7380785417592706e-12
37 3.6241144015164514e-15 1.894804621635921e-11 1.1475879338010407e-12
38 8.440496482114228e-16 1.7342220491446898e-12 3.4991369569619143e-13
39 5.787550502172918e-16 2.563995225280647 0.017093301502000863
40 3.7720170723508235e-16 4.1516794595023486e-13 5.485051200145938e-14
41 3.7720170723508235e-16 1.396601175925498e-13 1.5082798842133835e-14
42 1.9906155178342068e-16 3.288393728661437e-14 3.9069841175008566e-15
43 1.3974813004606267e-16 8.471252932393162e-15 1.4520178323249146e-15
44 9.625948083149312e-19 0.07531378711479737 0.0005020919140994443
45 9.625948083149312e-19 2.398145524207185e-15 4.255406165962221e-16
46 7.587451689091868e-19 1.1311459506593254e-15 2.20537527319186e-16
47 7.587451689091868e-19 8.06550569122518e-16 1.146615819128312e-16
48 5.002774718252979e-20 4.1168480850594117e-16 3.6914951492045416e-17
49 4.098786523876955e-20 8.734581466490933e-17 7.758960818537709e-18
50 2.859598660504208e-20 2.171082998804635e-17 2.0458750861865494e-18
51 1.3869137329193593e-20 4.1474897734134476e-18 6.857730014517774e-19
52 9.251252898778727e-22 2.805601715251568e-18 2.967337646197961e-19
53 6.510907657452313e-24 5.281482401218925e-19 8.784288624738157e-20
54 5.267200061908124e-22 2.0266336561016597e-19 4.020065664850527e-20
55 3.389472145736706e-22 1.3667207290967596 0.00911147152731173
56 9.699077094103364e-23 6.490994013191462e-20 6.181922356288426e-21
57 8.76431849438617e-23 1.9775176500911784e-20 1.808067261321071e-21
58 4.090679244790628e-23 5.237050417366387e-21 6.971491851941615e-22
59 1.6314356650228844e-23 1.5526377825606223e-21 3.04654684482566e-22
60 5.824507162045027e-25 10.476577577700876 0.06984385051800585
61 5.824507162045027e-25 7.457777401989094e-22 9.608099507709103e-23
62 9.16853587093121e-26 1.9057433330328194e-22 4.697843113021691e-23
63 1.8115401827495315e-26 1.204838533366387e-22 2.2150653911741324e-23
64 1.8115401827495315e-26 7.573280838009744e-23 1.0417514435199323e-23
65 4.971795855155427e-26 2.9975045957583906e-23 4.237207765057352e-24
66 8.144199985501726e-27 1.096013759430128e-23 1.2822766355129823e-24
67 4.894881916896378e-27 2.7434286698352597e-24 4.674124241280072e-25
68 3.332937324558775e-27 1.0022296362824754e-24 1.831090575154351e-25
69 1.072850831100576e-28 4.949194990220845e-25 8.859926122106967e-26
70 1.072850831100576e-28 2.524354896707238e-25 3.533905425143799e-26
71 1.072850831100576e-28 5.982168902562499e-26 1.2835776669904545e-26
72 8.914128228997433e-29 4.549518612590619e-26 7.187027717542759e-27
73 7.020862056467005e-29 2.590698098836325e-26 3.391397176708353e-27
74 3.549874073494553e-29 8.66169273932671e-27 1.4339282446383376e-27
75 3.1554436208840472e-30 4.153352665988627e-27 5.6755912594301065e-28
76 3.1554436208840472e-30 1.2274675685238944e-27 1.689003788805201e-28
77 0.0 3.6366487730688644e-28 8.637501004899932e-29
"""
