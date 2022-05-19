# !/usr/bin/env python3

"""OPTIMISATION CODE TO SEND TO CLUSTER"""

from lenia_package import *  # Load package

# Load creatures

# iter = float(Sys.getenv("PBS_ARRAY_INDEX"))

global g, s
g = 0.5  # set gradient count
s = 0  # set seed

def run_simulations(i):
    global s, g
    np.random.seed(s)

    orbium = Creature("orbium")  # Initiate creature
    obstacle = ObstacleChannel(n=1, r=60, gradient = g)

    optimise_timely(orbium, obstacle, N=100, run_time = 70)

    g += 1


for i in range(10):
    run_simulations(i)
