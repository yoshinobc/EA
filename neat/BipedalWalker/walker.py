from __future__ import print_function
import os
import neat
import visualize
import gym
import numpy as np
from gym import wrappers
import operator

env = gym.make("BipedalWalker-v2")
gen = 0
def eval_genomes(genomes, config):
    global gen
    total_reward_list = []
    gen += 1
    if 400 < gen <= 450:
        MAX_STEPS = 3700
    elif 350 < gen <= 400:
        MAX_STEPS = 3000
    elif 300 < gen <= 350:
        MAX_STEPS = 2500
    elif 250 < gen <= 300:
        MAX_STEPS = 1800
    elif 200 < gen <= 250:
        MAX_STEPS = 1200
    elif 150 < gen <= 200:
        MAX_STEPS = 800
    elif 100 < gen <= 150:
        MAX_STEPS = 500
    elif 50 < gen <= 100:
        MAX_STEPS = 300
    elif 0 <= gen <= 50:
        MAX_STEPS = 100

    for genome_id, genome in genomes:
        genome.fitness = 0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        total_reward = 0
        for i in range(MAX_STEPS):
            observation = env.reset()
            action = net.activate(observation)
            action = np.clip(action,-1,1)
            observation,reward,done,info = env.step(action)
            total_reward += reward
            #if (-4.8  > observation[0]) or (observation[0] > 4.8) or (0.017453292519943 < observation[3] < -0.017453292519943) or (episode_reward >= MAX_STEPS):
            if done:
                break

        genome.fitness = total_reward
        total_reward_list.append(total_reward)


    #genomes.sort(key = operator.attrgetter('fitness'))
    total_reward_list = total_reward_list.sort()
    print(total_reward_list)
    print(len(total_reward_list))
    fitness_mean = total_reward_list[75]
    for genome in genomes:
        genome.fitness = (genome.fitness - fitness_mean)**2,

def run(config_file):
    global env
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)
    #p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-195')
    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5))

    # Run for up to 300 generations.

    winner = p.run(eval_genomes, 200)


    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    # Show output of the most
    #fit genome against training data.
    print('\nOutput:')
    #winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    visualize.draw_net(config, winner, True)
    visualize.plot_stats(stats, ylog=False, view=True)
    visualize.plot_species(stats, view=True)
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    final_reward = 0
    #env = wrappers.Monitor(env, '/mnt/c/Users/bc/Documents/EA/neat/BipedalWalker/movies', force=True)
    observation = env.reset()
    while True:
        action = winner_net.activate(observation)
        action = np.clip(action,-1,1)
        observation,reward,done,info = env.step(action)
        final_reward += reward
        if done:
            print("final_reward :",final_reward)
            break

    """
    for n, g in enumerate([winner]):
        name = 'winner-{0}'.format(n)
        with open(name+'.pickle', 'wb') as f:
            pickle.dump(g, f)

        visualize.draw_net(config, g, view=False, filename=name + "-net.gv")
        visualize.draw_net(config, g, view=False, filename="net-enabled.gv",show_disabled=False)
        visualize.draw_net(config, g, view=False, filename="-net-enabled-pruned.gv",show_disabled=False, prune_unused=True)
    p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-4')
    p.run(eval_genomes, 10)
    """

if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.

    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward')
    run(config_path)
