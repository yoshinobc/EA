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

    def push_seq(self,state):
        self.seq = state

    def run(self,train=True):
        self.env.reset()
        self.reset_seq()
        total_reward=0

        for i in range(100):
            old_seq = self.seq.copy()

            action = self.agent.get_action(old_seq,train)

            observation,reward,done,info = self.env.step(action)
            total_rewrd +=reward

            state = observation
            self.puth_seq(state)
            new_seq = self.seq.copy()
