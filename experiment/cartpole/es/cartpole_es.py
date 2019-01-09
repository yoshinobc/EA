from deap import creator
from deap import base
from deap import tools
import numpy as np
import random
import gym
from deap import cma
import deap.algorithms
from NN import NNN
from gym import wrappers
# import cma.evolution_strategy
# from rnn2 import Basic_rnn, FullyConnectedRNN
# from pprint import pprint



# ------------------------------------------------------------------------------
#                        SET UP: OpenAI Gym Environment
# ------------------------------------------------------------------------------

ENV_NAME = 'CartPole-v2'
EPISODES = 1  # Number of times to run envionrment when evaluating
STEPS = 5000  # Max number of steps to run run simulation
count = 0
env = gym.make(ENV_NAME)
POPULATION_SIZE = 40
CROSS_PROB = 0.5
NGEN = 300   # Number of generations
MUTATION_RATE = 0.4 # PERCENTAGE OF GENES TO MUTATE
NUM_PARAMS = 572
N = 26
nn = NNN()

toolbox = base.Toolbox()
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)
def decide_weight(network):
    w_list = []
    if not network :
        w_list.extend(np.reshape(np.random.normal(0,1,(4,3)),(12)))
        w_list.extend(np.reshape(np.random.normal(0,1,(1,3)),(3)))
        w_list.extend(np.reshape(np.random.normal(0,1,(3,2)),(6)))
        w_list.extend(np.reshape(np.random.normal(0,1,(1,2)),(2)))
        w_list.extend(np.reshape(np.random.normal(0,1,(2,1)),(2)))
        w_list.extend(np.reshape(np.random.normal(0,1,(1,1)),(1)))
    return w_list

toolbox.register("attr_floats", decide_weight, nn.network)

toolbox.register("individual", tools.initIterate, creator.Individual,
                 toolbox.attr_floats)

toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def get_action(observation,individual,network):
    return nn.conclusion(observation,individual,network)

def evaluate(individual, MAX_REWARD=0 ):
    total_reward = 0
    network = nn.update(individual);

    for episode in range(EPISODES):
        observation = env.reset()
        for step in range(STEPS):
            action = get_action(observation,individual,network)
            env.render()
            observation, reward, done, _ = env.step(action)
            total_reward += reward

            if done:
                break
    total_reward /= EPISODES
    return [total_reward]


toolbox.register("evaluate", evaluate)



stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean, axis=0)
stats.register("std", np.std, axis=0)
stats.register("min", np.min, axis=0)
stats.register("max", np.max, axis=0)

logbook = tools.Logbook()
logbook.header = ['gen', 'nevals'] + stats.fields


def main():

    stats = tools.Statistics(lambda ind:ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    #pop,log = algorithms.cma.Strategy(pop,toolbox,cxpb=0.5,mutpb=0.2,ngen=10000,stats=stats,halloffame=hof,verbose=True)

    np.random.seed(64)

    # The CMA-ES algorithm
    strategy = cma.Strategy(centroid=[5.0]*N, sigma=3.0,_lambda=250)
    toolbox.register("generate",strategy.generate,creator.Individual)
    toolbox.register("update", strategy.update)

    halloffame = tools.HallOfFame(1)

    halloffame_array = []
    C_array = []
    centroid_array = []
    for i in range(NGEN):
        # 新たな世代の個体群を生成
        population = toolbox.generate()
        # 個体群の評価
        fitnesses = toolbox.map(toolbox.evaluate, population)
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit

        fits = [ind.fitness.values[0] for ind in population]

        halloffame.update(population)
        length = len(population)
        mean = sum(fits) / length
        sum2 = sum(x*x for x in fits)
        std = abs(sum2 / length - mean**2)**0.5

        #print("gen:",i,"  Min %s" % min(fits),"  Max %s" % max(fits),"  Avg %s" % mean,"  Std %s" % std)
        print(i,max(fits),mean)

        with open('CartPole-v2.txt',mode='a') as f:
            f.write(str(i))
            f.write(" ")
            f.write(str(max(fits)))
            f.write(" ")
            f.write(str(mean))
            f.write("\n")

        # 個体群の評価から次世代の計算のためのパラメタ更新
        
        toolbox.update(population)

        # hall-of-fameの更新
        halloffame.update(population)

        halloffame_array.append(halloffame[0])
        C_array.append(strategy.C)
        centroid_array.append(strategy.centroid)
        record = stats.compile(population)
        logbook.record(gen=i, nevals=1, **record)
        if i % 100 == 0 :
            print('generation',i,":",record)

if __name__ == "__main__":
    #env = wrappers.Monitor(env,'/mnt/c/Users/bc/Documents/EA/BipedalWalker_ES/movies/', video_callable=(lambda ep: ep % 8000 == 0),force=True)
    pop, logbook, hof = main()
    print(hof.items[0].fitness.values[0])
    print(logbook)

