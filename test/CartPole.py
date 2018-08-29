import numpy as np
from deap import algorithms,base,creator,tools,gp
import random
import operator
import math
import gym
from gym import wrappers
import time
import  networkx as nx
import matplotlib.pyplot as plt

ENV = gym.make("CartPole-v0")
MAX_STEPS=200
NGEN=40

def safeDiv(left,right):
    if(right==0):
        return 0
    try:
        return left/right
    except ZeroDivisionError:
        return 0

pset = gp.PrimitiveSet("MAIN", 4)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(safeDiv,  2)
pset.addPrimitive(operator.neg, 1)
pset.addPrimitive(math.cos, 1)
pset.addPrimitive(math.sin, 1)
pset.addEphemeralConstant("rand101", lambda:random.randint(-1,1))
"""
pset.renameArguments(ARG0='x')
pset.renameArguments(ARG1='v')
pset.renameArguments(ARG2='theta')
pset.renameArguments(ARG3='av')
"""
pset.renameArguments(ARG0='theta')
pset.renameArguments(ARG1='av')
pset.renameArguments(ARG2='x')
pset.renameArguments(ARG3='v')
'''
pset.addEphemeralConstant("theta2", lambda:ARG0*2)
pset.addEphemeralConstant("theta3", lambda:ARG0*3)
pset.addEphemeralConstant("theta4", lambda:ARG0*4)
'''
creator.create("FitnessMax",base.Fitness,weights=(1.0,))
creator.create("Individual",gp.PrimitiveTree,fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population",tools.initRepeat,list,toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

def EV(individual):
        observation = ENV.reset()
        episode_reward=0
        for step in range(MAX_STEPS):
            action = get_action(observation,individual)

            observation_next,reward,done,info_not = ENV.step(action)
            if done:
                ENV.render()
                break

            observation = observation_next
            episode_reward += reward
            #if Myclass.count>=2700:
            #    env = wrappers.Monitor(env,'./movie/cartpole-experiment-1')
            ENV.render()
        Myclass.count+=1
        return episode_reward,

toolbox.register("evaluate", EV)
toolbox.register("mate", gp.cxOnePoint)
#toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb = 0.1)
toolbox.register("select",tools.selTournament,tournsize=3)
toolbox.register("expr_mut", gp.genFull, min_= 0, max_ = 2)
toolbox.register("mutate",gp.mutUniform,expr=toolbox.expr_mut,pset=pset)


def get_action(observation,individual):
    action = decide_action(observation,individual) ##
    return action
    #observation[カートの位置，カートの速度，棒の角度，棒の角速度]
def decide_action(observation,individual):
    func = toolbox.compile(expr=individual)
    if(func(observation[2],observation[3],observation[0],observation[1]) >= 5):
        return 1
    else :
        return 0
class Myclass:
    count=0
    FT_avg=[]
    FT_max=[]
    gen=[]
    for i in range(1,NGEN+1):
        gen.append(i)
def main():
        random.seed(1)
        pop = toolbox.population(n=150)
        hof = tools.HallOfFame(1)

        stats_fit= tools.Statistics(lambda ind: ind.fitness.values)
        stats_size = tools.Statistics(len)

        mstats = tools.MultiStatistics(fitness=stats_fit,size=stats_size)
        mstats.register("avg", np.mean)
        mstats.register("std", np.std)
        mstats.register("min", np.min)
        mstats.register("max", np.max)
        CXPB=0.5
        MUTPB=0.1
        pop,log = algorithms.eaSimple(pop, toolbox, CXPB, MUTPB, NGEN, stats = mstats, halloffame=hof,verbose=True)
        return pop, log, hof

if __name__=='__main__':
    pop,log,hof = main()
    expr = tools.selBest(pop,1)[0]
    print(expr)
    nodes, edges, labels = gp.graph(expr)
    print(nodes)
    print(edges)
    print(labels)
    '''
    g = nx.Graph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    pos = nx.drawing.nx_agraph.graphviz_layout(g,prog="dot")
    print(pos)
    nx.draw_networkx_nodes(g,pos)
    nx.draw_networkx_edges(g, pos)
    nx.draw_networkx_labels(g,pos,labels)
    plt.show()
    '''
