import numpy

from deap import algorithms
from deap import base
from deap import benchmarks
from deap import cma
from deap import creator
from deap import tools

import matplotlib.pyplot as plt

N = 2  # 問題の次元
NGEN = 100  # 総ステップ数

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("evaluate", benchmarks.himmelblau)

def map(gen):
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
    sum_min = 0
    # The CMA-ES algorithm
    strategy = cma.Strategy(centroid=[5.0]*N, sigma=3.0, lambda_=150)
    toolbox.register("generate", strategy.generate, creator.Individual)
    toolbox.register("update", strategy.update)

    halloffame = tools.HallOfFame(1)

    halloffame_array = []
    C_array = []
    centroid_array = []
    x = numpy.arange(-6, 6, 0.05) #x軸の描画範囲の生成。0から10まで0.05刻み。
    y = numpy.arange(-6, 6, 0.05) #y軸の描画範囲の生成。0から10まで0.05刻み。

    X, Y = numpy.meshgrid(x, y)

    print("gen ","min ","max ","mean ","std")

    #plt.pcolormesh(X, Y, Z,cmap='hsv') # 等高線図の生成。cmapで色付けの規則を指定す
    #pp=plt.colorbar (orientation="vertical") # カラーバーの表示
    #pp.set_label("Label", fontname="Arial", fontsize=24) #カラーバーのラベル
    for gen in range(NGEN):
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        Z = numpy.power((numpy.power(X,2) + Y - map(gen)),2) + numpy.power((X + numpy.power(Y,2) - map(gen)),2)
        ax.set_xlim([-6,6])
        ax.set_ylim([-6,6])
        plt.pcolormesh(X, Y, Z, cmap='hsv') # 等高線図の生成。cmapで色付けの規則を指定する。
        def func(pop):
            pop = numpy.clip(pop,-6,6)
            X = pop[0]
            Y = pop[1]
            return numpy.power((numpy.power(X,2) + Y - map(gen)),2) + numpy.power((X + numpy.power(Y,2) - map(gen)),2)   # 表示する計算式の指定。等高線はZに対して作られる。
        #Z = (1 - 1/(1 + 0.05 * (np.power(X,2)+(np.power((Y - 10),2)))) - 1/(1 + 0.05*(np.power((X - 10),2) + np.power(Y,2))) - 1/(1 + 0.03*(np.power((X + 10),2) + np.power(Y,2))) - 1/(1 + 0.05*(np.power((X - 5),2) + np.power((Y + 10),2))) - 1/(1 + 0.1*(np.power((X + 5),2) + np.power((Y + 10),2))))*(1 + 0.0001*np.power((np.power(X,2) + np.power(Y,2)),1.2))


        plt.xlabel('X', fontsize=24)
        plt.ylabel('Y', fontsize=24)

        # 新たな世代の個体群を生成
        population = toolbox.generate()
        # 個体群の評価

        for ind in population:
            ax.scatter(ind[0],ind[1],c='blue',marker='.')

        fitnesses = toolbox.map(func, population)
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit,

        fits = [ind.fitness.values[0] for ind in population]

        length = len(population)
        mean = sum(fits) / length
        sum2 = sum(x*x for x in fits)
        std = abs(sum2 / length - mean**2)**0.5
        print(gen ,min(fits) ,max(fits) ,mean,std)
        sum_min += min(fits)
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
        #plt.closed()
    print(sum_min / 100)
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

"""
dynamic_cma_es.py
gen  min  max  mean  std
0 1.795909212562103 3528.0 1507.6866960307366 1124.1267874901002
1 2.9287290202258927 3362.0 1404.9273541473763 930.784564364336
2 0.24289245548452826 3200.0 1100.2620662082156 818.3322132097576
3 0.6478156502310786 3042.0 382.737500723579 597.5508566729409
4 0.5611028373564249 282.1862460454298 32.6430101321939 44.15562750440161
5 0.36918448468110393 110.85800643057516 24.61249579154825 23.589091378575805
6 0.373378944683083 762.6658073441096 101.50843826364056 151.35715219538088
7 0.8870385135496799 482.24372181075654 59.19561227406323 97.62011214207797
8 0.43650437957297744 416.00945141231944 46.15839922364911 71.38646977016798
9 1.587809757829674 90.81457484954636 31.984680704042482 21.722829895530513
10 2.618646540059577 816.276009543215 194.7639504256287 278.4675213662087
11 0.8906927792653857 129.97047668844215 36.925462539971164 29.385233175986983
12 0.06572569329365399 722.3846378617318 85.88472850384872 149.60141297851334
13 2.9581104816857455 179.69790406293046 41.36272572079473 36.37011618361842
14 0.9787539113575541 138.10492093030834 22.97524649393211 24.623625122950457
15 1.4266692643410894 111.24792112115699 29.604229889912148 25.824877540722547
16 0.10592830495174164 173.80773439597576 28.434304632079318 38.396568679130866
17 0.1992441327103746 54.97148059019938 13.074447140156696 11.886258746235926
18 0.05655607935062663 41.504465612965575 10.068644076005269 8.361265488808675
19 0.22387994203055514 53.19502470067461 12.363069303096449 11.564981581394308
/home/bc/.local/lib/python3.5/site-packages/matplotlib/pyplot.py:523: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
  max_open_warning, RuntimeWarning)
20 0.6857934752653594 115.17994468621634 21.615666306887725 22.504139422308207
21 0.019780175709869525 115.3828588282062 22.2286772325836 22.821228755859313
22 0.06749348424956289 82.68980092381211 21.506217899367133 22.545899041771907
23 0.01789116391565779 63.22406166575598 13.53515580938497 15.284580804635956
24 0.12666535927519101 26.60794156791476 4.513386140362217 5.468545215459991
25 0.05485836209068796 25.397508685137964 6.394225265380379 6.718063667701555
26 0.18068859811959442 37.00279681100683 6.8963113347406075 6.905026924635172
27 0.16041354600321522 40.02744458852435 7.4361591795879916 10.034115325555733
28 0.300229496707507 72.33484309481707 8.648799240269621 12.86114158803848
29 0.12243756669337624 50.71922116199468 14.477065113136131 15.54056034286982
30 0.012073928016331354 97.93008576969433 14.43177158470028 20.963328870801732
31 0.005644458854762826 105.79205331686069 16.953874471777436 24.404277301338748
32 0.08107337644408683 74.07232960478137 18.16534289255164 21.91026288998727
33 0.14264512689542153 80.70253155035664 17.330621129646882 21.184394986703936
34 0.2090923662101905 87.2335068178925 16.948838349989167 21.116870274850818
35 0.037138754314178415 104.45168184468123 17.056079051233144 24.15492355339111
36 0.0037763485759602505 54.9383676404547 8.46800909449941 12.415861674205729
37 0.04093990394986042 103.93492740587838 13.297323596056653 21.114853066030786
38 0.00020496039906609708 47.14064224569024 8.56250602966857 11.915265167292436
39 0.026429595704455838 51.17534463237189 11.153916551760632 13.664000155359494
40 0.03697804453262083 87.96849168804083 13.62746244938172 22.099808787592657
41 0.006431025603562807 62.91870290761763 14.320499208043149 17.172732806454775
42 0.08333853737741642 196.7383352647356 18.28659338000347 35.53427331802004
43 0.002624653831782374 109.13135559401535 12.913080260930457 24.736903386379073
44 0.028732203166462564 139.96746764834245 16.115499909289724 29.361546153588332
45 0.0030845066754517773 573.63079194502 31.5063230954075 95.1175984968856
46 0.0007174403689534544 522.2650257652341 22.42083921493363 83.594033307117
47 0.07787812649609305 10.094373971855532 5.430532910246017 3.6434334855691297
48 0.04367392899205255 611.7188219085518 28.722837518050227 94.05150217771128
49 0.7221507500984111 35.80796735760638 15.938090319483525 10.97041166064791
50 0.31567349315490745 1634.490023876408 150.1598692623456 331.4761112527313
51 0.014015905362220337 435.16331213580565 63.04565647040654 81.07006075826322
52 0.42212779556544333 293.56225369030994 31.476633112880393 48.00356319759723
53 0.0656575838220597 45.59370203762227 16.02553549327493 13.823991336469968
54 0.3048131370485402 68.04145380699485 13.782276610811934 15.133340912411166
55 0.07848921493593997 102.55952841004863 23.029539525798405 24.39522235537944
56 0.06270234834862806 116.99238610903794 16.80229710621865 24.431338981993637
57 0.051605218229396635 108.44114928369885 15.33741566256883 22.200593193991303
58 0.2249954859767844 86.08749186067199 20.485643575102667 20.777596514562543
59 0.2552742073990701 59.02084292696599 16.159191106113372 15.076646907907753
60 0.34716917414593373 132.0600060548811 18.380326692186255 25.472554390535507
61 0.21947673687376382 91.45042509177034 21.449405688271888 23.4184298082348
62 0.0035806726063493327 93.1073191100972 19.661319138217777 22.98134205955433
63 0.08102240482420915 85.7897447011825 19.526130739358404 23.219423515653094
64 0.19456832467308177 55.78921776469061 10.532251734225694 11.7334800519505
65 0.07688094695234689 119.32244378139725 14.151660511835843 21.576684381665483
66 0.002216709310225856 61.469617283592235 11.333702952315365 13.659526525272458
67 0.04236835040771383 50.16586128106418 12.841787056763982 12.632740101618607
68 0.039758511286856923 39.54615388400423 8.209188507005805 9.839456265062525
69 0.025281264218678566 61.47418037337888 8.516613250057663 11.997970679249413
70 0.00957964270814678 38.59636822844955 7.536376023084057 8.399214134049847
71 0.006193272324631687 68.7067880044108 10.179463472726024 15.514277440309888
72 0.0975144472379068 49.45618478216816 7.593622239001829 8.804099112332183
73 0.09188892536499739 74.19207953971471 10.236259221068549 13.813678822230395
74 0.054680022026152915 75.95912928545097 11.408518485446447 15.700738819831495
75 0.027580823028587425 40.95367694645618 8.84832568152311 11.043058736896269
76 0.03754735698709165 91.6573829433467 15.069115332684586 20.95061101144249
77 0.043800367908949665 41.79716812877555 11.149183595206214 11.253039998957751
78 0.11222654846155992 72.65922396132436 10.304550353060485 16.484547837063676
79 0.002518652935700817 74.69797252586116 17.137000300592884 20.0658324472898
80 0.019028103986337766 199.69328000833013 20.08121659761997 38.65503992143738
81 0.02719529347853946 54.08122414184551 10.693116896248757 13.507714852453699
82 0.013591511244227241 80.17352310883874 11.415500502820247 15.424028200073538
83 0.023507592800447973 59.73737051104304 10.584332573385947 11.63948199491699
84 0.04312585046706789 210.29673049123087 16.32854933737972 35.222837454902
85 0.03531506519317224 105.29737743089095 17.818153155322797 25.721623738351784
86 0.0013750833347833877 108.62982709385781 13.266723998845046 22.28502057324108
87 0.0010079022579773742 110.48668020047839 13.487216465240238 24.180711093408334
88 0.024056599443395152 225.89252823764974 14.761859073583675 38.36138243645336
89 0.008229459221509116 213.69700070804188 10.802522994465019 34.61282910139313
90 0.02282720692071552 3.2565267008978203 2.1038599996668066 0.9845790065423367
91 0.07949123904183503 86.66818943773973 9.618430821322388 17.276532333769982
92 0.21636514049651903 44.30499133213546 11.474288325464435 8.697347672887297
93 0.020659321400120778 77.12554617402522 23.14277098191898 17.784697830583017
94 0.7772739190838774 318.3087427562499 43.3811118119372 53.84962121675634
95 0.6529835011153329 498.5594851455063 54.942647387600104 92.41911034508081
96 0.16962297929526063 83.24245845744426 16.346995715822494 17.954935850885146
97 0.04750347275886526 105.10060919041354 18.453319982604132 22.180190803069827
98 0.48502167757462605 62.404491045967376 14.734072411976555 14.851076928227636
99 0.6402988895293666 59.43431364284821 12.99645139631447 14.644765204131168
100 0.1129834359521869 76.31285744329195 20.87142895693461 19.2954901008838
"""
