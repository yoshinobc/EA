import deap
#import NN
#from NN import NNN
from deap import base
from deap import creator
from deap import tools
import math
import gym
import random
from gym import wrappers
import time
import numpy as np
from deap import algorithms
from skimage.color import rgb2gray
from skimage.transform import resize

#nn = NNN()
ENV = gym.make("AsteroidsNoFrameskip-v4")
MAX_STEPS = 201
w_list = []
# In[9]:

creator.create("FitnessMax",base.Fitness,weights=[1.0])
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

def decide_weight(network):
    #print(np.reshape(network['W1'],(12)))
    w_list = []
    if not network :
        w_list.extend(np.reshape(np.random.normal(0,1,(4,6)),(24)))
        w_list.extend(np.reshape(np.random.normal(0,1,(1,6)),(6)))
        w_list.extend(np.reshape(np.random.normal(0,1,(6,3)),(18)))
        w_list.extend(np.reshape(np.random.normal(0,1,(1,3)),(3)))
        w_list.extend(np.reshape(np.random.normal(0,1,(3,1)),(3)))
        w_list.extend(np.reshape(np.random.normal(0,1,(1,1)),(1)))

    return w_list

toolbox.register("gene",decide_weight,nn.network)
toolbox.register("individual",tools.initIterate,creator.Individual,toolbox.gene)
toolbox.register("population",tools.initRepeat,list,toolbox.individual)

def get_initial_state(observation,last_observation):
    STATE_LENGTH = 4
    FRAME_WIDTH = 84
    FRAME_HEIGHT = 84
    processed_observation = np.maximum(observation,last_observation)
    processed_observation = np.uint8(resize(rg2gray(processed_observation), (FRAME_WIDTH,FRAME_HEIGHT)) * 255)
    state = [processed_observation for _ in xrange(STATE_LENGTH)]
    return np.stack(state, axis=0)

def preprocess(observation,last_observation):
    processed_observaton = np.maximum(observation,last_observation)
    processed_observation = np.uint8(resize(rgb2gray(processed_observation), (FRAME_WIDTH, FRAME_HEIGHT)) * 255)
    return np.reshape(processed_observation, (1, FRAME_WIDTH, FRAME_HEIGHT))

def EV():
    terminal = False
    observation = env.reset()
    for _ in xrange(random.randint(1,NO_OP_STEPS)):
        last_observation = observation
        observation, _, _, _ = env.step(0)
    state = get_initial_state(observation,last_observation)
    while not terminal:
        last_observation = observation
        action = get_action(state)
        observation,reward,terminal,_ = env.step(action)
        env.render()
        processed_observation = preprocess(observation, last_observation)
        state = run(state,action,terminal,processed_observation)

def get_action(state):
    print(state)

toolbox.register("evaluate",EV)
toolbox.register("mate",tools.cxBlend,alpha=0.5) #float
toolbox.register("mutate",tools.mutGaussian,mu=0,sigma=0.5,indpb=0.05) #mutFllipBit floatに対して津えるやつ
toolbox.register("select",tools.selLexicase)

def get_action(observation,individual,network):
    action = decide_action(observation,individual,network)
    return action

def decide_action(observation,individual,network):
    return nn.conclusion(observation,individual,network)

def main():
    random.seed(1)

    pop = toolbox.population(n=150)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind:ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    pop,log = algorithms.eaSimple(pop,toolbox,cxpb=0.5,mutpb=0.2,ngen=40,stats=stats,halloffame=hof,verbose=True)

    return pop,log,hof
if __name__=='__main__':
    print("pop_num = ",150)
    print("gen_num ",40)
    pop,log,hof = main()
    best_ind = tools.selBest(pop,1)[0]
    for ind in hof :
        print(ind ,end="")
        print(ind.fitness)
    print(nn.update(best_ind))
