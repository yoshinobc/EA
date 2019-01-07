import numpy as np
from deap import algorithms,base,cma,creator,tools,benchmarks
import matplotlib.pyplot as plt
import random
import csv
import time

def func(pop):
      return (1 - (1 / (1 * np.linalg.norm(np.array(pop) - np.array([-2,-2]), axis=0) + 1))) + (1 - (1 / (2 *np.linalg.norm(np.array(pop) - np.array([4,4]),  axis=0) + 1))),


print(func([4,4]))
