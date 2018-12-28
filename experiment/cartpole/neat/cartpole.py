from __future__ import print_function
import os
import neat
import visualize
import gym
import numpy as np
from gym import wrappers
import pickle

MAX_STEPS = 200
env = gym.make("CartPole-v1")
count = 0
reward_list = []
#env = wrappers.Monitor(env, '/mnt/c/Users/bc/Documents/EA/neat/cartpole/movies', video_callable=(lambda ep: ep % 150 == 0), force=True)
def eval_genomes(genomes, config):
    global env
    global MAX_STEPS
    global count
    max_episode_fitness = -100
    for genome_id, genome in genomes:
        genome.fitness = 0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        observation = env.reset()
        episode_reward = 0
        for _ in range(10):
            for i in range(1000):
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

        genome.fitness = episode_reward / 10

        if max_episode_fitness <= episode_reward / 10:
            max_episode_fitness = episode_reward / 10
            winner = genome
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
    reward_list = []
    #for j in range(20):
        # Run for up to 300 generations.
    winner = p.run(eval_genomes, 20)

        # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

        # Show output of the most fit genome against training data.
    print('\nOutput:')
        #winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
        
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
