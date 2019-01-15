from __future__ import print_function
import os
import neat
import visualize
import gym
import numpy as np
from gym import wrappers
import pickle

MAX_STEPS = 200
env = gym.make("CartPole-v2")
count = 0
reward_list = []
#env = wrappers.Monitor(env, '/mnt/c/Users/bc/Documents/EA/neat/cartpole/movies', video_callable=(lambda ep: ep % 150 == 0), force=True)
def eval_genomes(genomes, config):
    global env
    global MAX_STEPS
    global count
    global reward_list

    max_episode_fitness = -100
    for genome_id, genome in genomes:
        genome.fitness = 0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        observation = env.reset()
        episode_reward = 0
        for _ in range(3):
            for i in range(5000):
                action = net.activate(observation)
                if action[0] > 0.5:
                    action = 1
                else :
                    action = 0
                observation,reward,done,info = env.step(action)
                episode_reward += reward
            #if (-4.8  > observation[0]) or (observation[0] > 4.8) or (0.017453292519943 < observation[3] < -0.017453292519943) or (episode_reward >= MAX_STEPS):
                if done:
                    observation = env.reset()
                    break

        genome.fitness = episode_reward / 3

        if max_episode_fitness <= episode_reward / 3:
            max_episode_fitness = episode_reward / 3
            winner = genome
    """
    print(max_episode_fitness)
    winner_net = neat.nn.FeedForwardNetwork.create(genome, config)
    env = wrappers.Monitor(env, '/home/bc/Documents/EA/experiment/cartpole/neat/movies', force=True)
    observation = env.reset()
    for i in range(500):
            action = winner_net.activate(observation)
            env.render()
            if action[0] > 0.5:
                action = 1
            else :
                action = 0
            observation,reward,done,info = env.step(action)
            episode_reward += reward
            #if (-4.8  > observation[0]) or (observation[0] > 4.8) or (0.017453292519943 < observation[3] < -0.017453292519943) or (episode_reward >= MAX_STEPS):
            if done:
                observation = env.reset()
                break
    """
    reward_list.append(episode_reward)


    for n, g in enumerate([winner]):
        name = 'winner-{0}'.format(n)
        with open(name+'.pickle', 'wb') as f:
            pickle.dump(g, f)

        visualize.draw_net(config, g, view=False, filename=str(count)+name + "-net.gv")
        visualize.draw_net(config, g, view=False, filename=str(count)+"net-enabled.gv",show_disabled=False)
    count +=1

def run(config_file):
    global env
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(1))
    #for j in range(20):
        # Run for up to 300 generations.
    winner = p.run(eval_genomes, 300)
    print(reward_list)
        # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

        # Show output of the most fit genome against training data.
    print('\nOutput:')
    visualize.draw_net(config, winner, True)
    visualize.plot_stats(stats, ylog=False, view=True)
    visualize.plot_species(stats, view=True)
    visualize.draw_net(config,winner,view=True,filename="winner-feedforward-evabled-pruneg.gv",show_disabled=False,prune_unused=True)
        #winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    for n, g in enumerate([winner]):
        visualize.draw_net(config, g, view=False, filename=str(j)+"-net-enabled-pruned.gv",show_disabled=False, prune_unused=True)
    #p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-4')
    #p.run(eval_genomes, 10)

if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.

    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-Feedforward')
    run(config_path)
"""
Best individual in generation 299 meets fitness threshold - complexity: (2, 6)

Best genome:
Key: 1405
Fitness: 5000.0
Nodes:
    0 DefaultNodeGene(key=0, bias=-1.2698343989655831, response=2.6988906580946357, activation=sigmoid, aggregation=sum)
    331 DefaultNodeGene(key=331, bias=-1.4451567783300336, response=0.5157602848669417, activation=sigmoid, aggregation=sum)
Connections:
    DefaultConnectionGene(key=(-4, 0), weight=3.5233273298611456, enabled=True)
    DefaultConnectionGene(key=(-3, 0), weight=3.5912707244991986, enabled=True)
    DefaultConnectionGene(key=(-2, 0), weight=0.37479301128694964, enabled=True)
    DefaultConnectionGene(key=(-2, 331), weight=1.8610016072037239, enabled=True)
    DefaultConnectionGene(key=(-1, 331), weight=1.1287191925892466, enabled=True)
    DefaultConnectionGene(key=(331, 0), weight=1.459582238427499, enabled=True)
"""

"""
[29.0, 208.0, 504.0, 367.0, 27.0, 665.0, 330.0, 45.0, 1186.0, 1172.0, 1483.0, 1159.0, 1218.0, 29.0, 29.0, 33.0, 51.0, 29.0, 216.0, 51.0, 46.0, 55.0, 119.0, 29.0, 28.0, 27.0, 26.0, 15000.0, 622.0, 31.0, 62.0, 15000.0, 4250.0, 1301.0, 1517.0, 28.0, 279.0, 15000.0, 29.0, 15000.0, 1536.0, 31.0, 437.0, 1493.0, 15000.0, 958.0, 1449.0, 303.0, 402.0, 15000.0, 15000.0, 116.0, 4984.0, 149.0, 1098.0, 12408.0, 376.0, 203.0, 294.0, 15000.0, 208.0, 15000.0, 168.0, 308.0, 27.0, 15000.0, 28.0, 161.0, 937.0, 29.0, 1387.0, 1276.0, 281.0, 1628.0, 10987.0, 28.0, 1554.0, 417.0, 15000.0, 226.0, 28.0, 217.0, 161.0, 515.0, 1852.0, 459.0, 356.0, 518.0, 5644.0, 1560.0, 29.0, 356.0, 1403.0, 3178.0, 27.0, 1462.0, 198.0, 862.0, 5581.0, 15000.0, 15000.0, 1440.0, 60.0, 1502.0, 15000.0, 15000.0, 1593.0, 24.0, 15000.0, 1263.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 11328.0, 15000.0, 15000.0, 300.0, 15000.0, 28.0, 30.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 3964.0, 30.0, 3433.0, 15000.0, 29.0, 27.0, 937.0, 15000.0, 908.0, 2225.0, 28.0, 15000.0, 26.0, 65.0, 39.0, 30.0, 96.0, 877.0, 15000.0, 28.0, 15000.0, 28.0, 15000.0, 26.0, 10045.0, 15000.0, 546.0, 15000.0, 15000.0, 15000.0, 38.0, 30.0, 88.0, 45.0, 194.0, 15000.0, 15000.0, 93.0, 15000.0, 15000.0, 226.0, 15000.0, 3665.0, 288.0, 1783.0, 678.0, 15000.0, 15000.0, 30.0, 15000.0, 178.0, 15000.0, 15000.0, 29.0, 15000.0, 28.0, 394.0, 30.0, 29.0, 15000.0, 15000.0, 31.0, 15000.0, 15000.0, 37.0, 15000.0, 15000.0, 15000.0, 1186.0, 10336.0, 5949.0, 15000.0, 1734.0, 30.0, 15000.0, 521.0, 1447.0, 7208.0, 10813.0, 53.0, 36.0, 29.0, 30.0, 30.0, 15000.0, 49.0, 41.0, 211.0, 27.0, 719.0, 255.0, 277.0, 28.0, 30.0, 15000.0, 120.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 27.0, 117.0, 28.0, 28.0, 10011.0, 15000.0, 705.0, 902.0, 355.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 1366.0, 29.0, 15000.0, 25.0, 444.0, 1370.0, 27.0, 15000.0, 1377.0, 15000.0, 1361.0, 15000.0, 575.0, 15000.0, 15000.0, 15000.0, 15000.0, 29.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0, 3083.0, 15000.0, 488.0, 8585.0, 15000.0, 30.0, 15000.0, 111.0, 15000.0, 15000.0, 15000.0, 31.0, 1858.0, 15000.0, 1860.0, 1953.0, 15000.0, 15000.0, 2017.0, 29.0, 2083.0, 328.0, 2166.0, 1981.0, 31.0, 15000.0, 15000.0, 25.0, 15000.0, 63.0, 15000.0, 29.0, 15000.0, 29.0]
"""