import gym
import random
import numpy as np
from skimage.color import rgb2gray
from skimage.transform import resize

env=gym.make("PongNoFrameskip-v4")
NO_OP_STEPS = 30
STATE_LENGTH = 4
FRAME_WIDTH = 84
FRAME_HEIGHT = 84
def get_initial_state(observation,last_observation):
    processed_observation = np.maximum(observation,last_observation)
    processed_observation = np.uint8(resize(rgb2gray(processed_observation), (FRAME_WIDTH,FRAME_HEIGHT)) * 255)
    state = [processed_observation for _ in range(STATE_LENGTH)]
    return np.stack(state, axis=0)

def preprocess(observation,last_observation):
    processed_observation = np.maximum(observation,last_observation)
    processed_observation = np.uint8(resize(rgb2gray(processed_observation), (FRAME_WIDTH, FRAME_HEIGHT)) * 255)
    return np.reshape(processed_observation, (1, FRAME_WIDTH, FRAME_HEIGHT))


def get_action(state):
    for x in state:
        length += len(x)  #取り出した要素の長さを足していく
        print('list1の長さは{}です'.format(length))

for episode in range(20):
    done = False
    observation = env.reset()
    for _ in range(random.randint(1,NO_OP_STEPS)):
        last_observation = observation
        observation, _, _, _ = env.step(0)
    state = get_initial_state(observation,last_observation)
    print(state)
    while not done:
        last_observation = observation
        action = get_action(state)
        observation,reward,done,_ = env.step(3)
        env.render()
        processed_observation = preprocess(observation, last_observation)
