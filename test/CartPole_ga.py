class GA:
    import deap
    from deap import base
    from deap import creator
    from deap import tools
    import math
    import gym
    from gym import wrappers
    import time
    import numpy as np
    ENV = gym.make("CartPole-v0")
    MAX_STEPS = 200
    NGEN = 30
    static w_list = []

    def ga():
        creator.create("FitnessMax",base.Fitness,weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        toolbox = base.Toolbox()

        def decide_weight():
            w_list.append(np.reshape(NN.network['W1'],(1,12)))
            w_list.append(np.reshape(NN.network['B1'],(1,3)))
            w_list.append(np.reshape(NN.network['W2'],(1,6)))
            w_list.append(np.reshape(NN.network['B2'],(1,2)))
            w_list.append(np.reshape(NN.network['W3'],(1,2)))
            w_list.append(np.reshape(NN.network['B3'],(1,1)))

        toolbox.register("weight",decide_weight)
        toolbox.register("individual",tools.init_Repeat,creator.individual,toolbox.weight,100)
        toolbox.register("population",tools.initRepeat,list,toolbox.individual)

        def EV(individual):
            reward = 0;
            observation = ENV.reset()
            episode_reward = 0
            for step in range(MAX_STEPS):
                action = get_action(observaton,individual)

                observation_next,reward,done,info_not = ENV.step(action)

                if done:
                    ENV.render()
                    break

                observation = observation_next
                episode_reward += reward

                ENV.render()

            return episode_reward

        toolbox.register("evaluate",EV)
        toolbox.register("mate",tools.cxTwoPoint)
        toolbox.register("mutate",tools.mutFlipBit,indpb=0.05)
        toolbox.register("select",tools.selTournament,tournsize=3)

        def get_action(observation,individual):
            action = decide_action(observation,individual)
            return action

        def decide_action(observation,individual):
            return NN,conclusion(observation,individual)

        def main():
            random.seed(101)
            pop = toolbox.population(n=300)
            hof = tools.HallOfFame(1)
            stats = tools.Statistics(lambda ind:ind.fitness.values)
            stats.register("avg", numpy.mean)
            stats.register("std", numpy.std)
            stats.register("min", numpy.min)
            stats.register("max", numpy.max)

            pop,log = algorithms.eaSimple(pop,toolbox,cxpb=0.5,mutpb=0.2,ngen=40,stats=stats,halloffame=hof,verbose=True)

            return pop,log,hof
