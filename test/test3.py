import operator
import math
import random
import numpy
import matplotlib.pyplot as plt
import networkx as nx
import pylab as plt
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp
def safeDiv(left,right):
    try:
        return left/right
    except ZeroDivisionError:
        return 0
IND_SIZE = 100
#引数の一つ目は関数，二つ目は引数の数，三つ目の引数はプログラムの入力の数
pset = gp.PrimitiveSet("MAIN",1)
pset.addPrimitive(operator.add,2)
pset.addPrimitive(operator.sub,2)
pset.addPrimitive(operator.mul,2)
pset.addPrimitive(safeDiv,2)
pset.addPrimitive(operator.neg,1)
pset.addPrimitive(math.cos,1)
pset.addPrimitive(math.sin,1)
pset.addEphemeralConstant("rand101",lambda:random.randint(-1,1))#ノードの終端で定数ではなく，乱数などの関数から生成される値を用いるときにつかう．今回の場合は-1,0,1
pset.renameArguments(ARG0='x')

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

def evalSymbReg(individual,points):
    #tree表現から関数への変換
    func = toolbox.compile(expr=individual)
    #推定式と真の式との平均平方誤差の計算
    sqerrors = ((func(x) - x**4 - x**3 - x**2 - x)**2 for x in points)
    return math.fsum(sqerrors)/len(points),
def target(x):
    return x**4 - x**3 - x**2 - x
toolbox.register("evaluate",evalSymbReg,points=[x/10. for x in range(-10,10)])
toolbox.register("select",tools.selTournament,tournsize=3)
toolbox.register("mate",gp.cxOnePoint)
toolbox.register("expr_mut",gp.genFull,min_=0,max_=2)
toolbox.register("mutate",gp.mutUniform,expr=toolbox.expr_mut,pset=pset)
def main():
    random.seed(318)

    pop = toolbox.population(n=300)
    hof = tools.HallOfFame(1)

    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", numpy.mean)
    mstats.register("std", numpy.std)
    mstats.register("min", numpy.min)
    mstats.register("max", numpy.max)

    pop, log = algorithms.eaSimple(pop, toolbox, 0.5, 0.1, 40, stats=mstats,
                                   halloffame=hof, verbose=True)
    # logの表示
    return pop, log, hof
if __name__ == "__main__":
    pop,log,hof = main()
    """
    import  networkx as nx
    expr = toolbox.individual()
    nodes, edges, labels = gp.graph(expr)
    g = nx.Graph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    pos = nx.drawing.nx_agraph.graphviz_layout(g,prog="dot")
    print(pos)
    nx.draw_networkx_nodes(g,pos)
    nx.draw_networkx_edges(g, pos)
    nx.draw_networkx_labels(g,pos,labels)
    plt.show()
    """
    """
    expr = tools.selBest(pop,1)[0]
    x = [x/10. for x in range(-100,100)]
    y_t = numpy.zeros(len(x))
    y_pred = numpy.zeros(len(x))
    func = toolbox.compile(expr)
    for i, x0 in enumerate(x):
        y_t[i]= target(x0)
        y_pred[i] = func(x0)
    plt.plot(x, y_t, label="target")
    plt.plot(x, y_pred, label="prediction")
    plt.legend()
    plt.show()
    """
