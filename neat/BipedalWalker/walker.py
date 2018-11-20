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

    MAX_STEPS = 3700
    total_reward_fit_list = []
    for genome_id, genome in genomes:
        genome.fitness = 0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        final_reward = 0
        total_total_reward = 0
        total_reward = 0
        for _ in range(2):
            total_novelty = 0
            action_lists = []
            observation = env.reset()
            for i in range(MAX_STEPS):
                action = net.activate(observation)
                action = np.clip(action,-1,1)
                observation,reward,done,xylists = env.step(action)
                action_lists.append(xylists)
                total_reward += reward
                if done:
                    break
            for i in range(len(action_lists) - 1):
                total_novelty += np.sqrt((action_lists[i+1][0] - action_lists[i][0])**2 + (action_lists[i+1][1] - action_lists[i][1])**2)
            final_reward += total_novelty
        total_reward_fit_list.append(total_reward/3)
        total_reward_list.append(final_reward)

        genome.fitness = final_reward
    print(str(gen),str(max(total_reward_fit_list)),str(sum(total_reward_fit_list)/150))
    
    with open('liner_novelty.txt',mode='a') as f:
            f.write(str(gen))
            f.write(str(max(total_reward_fit_list)))
            f.write(str(sum(total_reward_fit_list)/150))
            f.write("\n")
            f.close()

    gen += 1
    cpgenomes = []
    for genome_id, ind in genomes:
        cpgenomes.append([genome_id,ind])


    for genome_id, ind in genomes:
        t = ind.fitness
        ind.fitness = 0
        for genome_id2,ind2 in cpgenomes:
            ind.fitness += abs((t - ind2.fitness))


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

    winner = p.run(eval_genomes, 1000)


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
