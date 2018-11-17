from deap import creator
from deap import base
from deap import tools
import numpy as np
import random
import gym
from deap import cma
import deap.algorithms
from walker_NN import NNN
from gym import wrappers
# import cma.evolution_strategy
# from rnn2 import Basic_rnn, FullyConnectedRNN
# from pprint import pprint



# ------------------------------------------------------------------------------
#                        SET UP: OpenAI Gym Environment
# ------------------------------------------------------------------------------

ENV_NAME = 'BipedalWalker-v2'
EPISODES = 2  # Number of times to run envionrment when evaluating
STEPS = 3700  # Max number of steps to run run simulation
count = 0
env = gym.make(ENV_NAME)
POPULATION_SIZE = 40
CROSS_PROB = 0.5
NGEN = 5000   # Number of generations
DEME_SIZE = 3  # from either side
MUTATION_RATE = 0.4 # PERCENTAGE OF GENES TO MUTATE
NUM_PARAMS = 572
N = 572
nn = NNN()

toolbox = base.Toolbox()
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)
def decide_weight(network):
    w_list = []
    if not network :
        w_list.extend(np.reshape(np.random.normal(0,1,(24,16)),(384)))
        w_list.extend(np.reshape(np.random.normal(0,1,(1,16)),(16)))
        w_list.extend(np.reshape(np.random.normal(0,1,(16,8)),(128)))
        w_list.extend(np.reshape(np.random.normal(0,1,(1,8)),(8)))
        w_list.extend(np.reshape(np.random.normal(0,1,(8,4)),(32)))
        w_list.extend(np.reshape(np.random.normal(0,1,(1,4)),(4)))

    return w_list

toolbox.register("attr_floats", decide_weight, nn.network)

toolbox.register("individual", tools.initIterate, creator.Individual,
                 toolbox.attr_floats)

toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def get_action(observation,individual,network):
    return nn.conclusion(observation,individual,network)

def evaluate(individual, MAX_REWARD=0 ):
    global count
    total_reward = 0
    network = nn.update(individual);

    for episode in range(EPISODES):
        observation = env.reset()

        episode_reward = 0
        if count % 8000 == 0:
            for step in range(STEPS):
                action = get_action(observation,individual,network)
                observation, reward, done, _ = env.step(action)
                episode_reward += reward

                if done:
                    break
        else:
            for step in range(STEPS):
                action = get_action(observation,individual,network)
                observation, reward, done, _ = env.step(action)
                episode_reward += reward

                if done:
                    break
        count+=1
    total_reward += episode_reward


    total_reward /= EPISODES
    return [total_reward]


toolbox.register("evaluate", evaluate)
toolbox.register("crossover",tools.cxESBlend,alpha = 0.2 )
toolbox.register("mutate",tools.mutESLogNormal,c=1.0,indpb=0.5)



hof = tools.HallOfFame(3, np.array_equal)  # Store the best 3 individuals

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
    strategy = cma.Strategy(centroid=[5.0]*N, sigma=3.0,_lambda=40)
    toolbox.register("generate",strategy.generate,creator.Individual)
    toolbox.register("update", strategy.update)

    halloffame = tools.HallOfFame(1)

    halloffame_array = []
    C_array = []
    centroid_array = []
    for gen in range(NGEN):
        # 新たな世代の個体群を生成
        population = toolbox.generate()
        # 個体群の評価
        fitnesses = toolbox.map(toolbox.evaluate, population)
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit

        # 個体群の評価から次世代の計算のためのパラメタ更新
        toolbox.update(population)

        # hall-of-fameの更新
        halloffame.update(population)

        halloffame_array.append(halloffame[0])
        C_array.append(strategy.C)
        centroid_array.append(strategy.centroid)
        record = stats.compile(population)
        logbook.record(gen=gen, nevals=1, **record)
        if gen % 100 == 0 :
            print('generation',gen,":",record)
    # 計算結果を描画
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from matplotlib.patches import Ellipse
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    X = numpy.arange(-5.12, 5.12, 0.1)
    Y = numpy.arange(-5.12, 5.12, 0.1)
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
        ax.axis([-5.12, 5.12, -5.12, 5.12])
        plt.draw()
    plt.show(block=True)


if __name__ == "__main__":
    #env = wrappers.Monitor(env,'/mnt/c/Users/bc/Documents/EA/BipedalWalker_ES/movies/', video_callable=(lambda ep: ep % 8000 == 0),force=True)
    pop, logbook, hof = main()
    print(hof.items[0].fitness.values[0])
    print(logbook)
    #
    # # Best controller
    # print(hof.items[0])

    # Save best as CSV
    # For some reason saves under _practice package
    # np.savetxt("./bestController_fully_10nodes2000.csv", hof.items[0],
    #            delimiter=",")
    # agent.set_weights(hof.items[0])
    # np.savetxt("./bestController_bu.csv", hof.items[0], delimiter=",")
    # agent.set_weights(hof.items[0])
    #
    # agent.set_weights(np.loadtxt("./bestController_fully_10nodes2000.csv", delimiter=","))
    #
    #
    # from evaluate import test
    # print(test(agent=agent))
