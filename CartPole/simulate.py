import gym
import deap
import random
from deap import base,creator
from deap import tools

class simulate:
    def __init__(self,environment,agent):
        self.agent = agent
        self.env = environment

        self.num_seq=MyClass.STATE_NUM

    def reset_seq(self):
        self.seq = np.zeros(self.num_seq)
