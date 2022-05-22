# !/usr/bin/env python3

"""OPTIMISATION CODE TO SEND TO CLUSTER"""

from lenia_package import *  # Load package

# Load creatures

iter = float(Sys.getenv("PBS_ARRAY_INDEX"))

global g, s
g = 0.5  # set gradient count
s = 0  # set seed

def run_simulations(i):
    global s, g

    orbium = Creature("orbium", cluster=True)  # Initiate creature
    obstacle = ObstacleChannel(n=1, r=60, gradient = g)

    optimise_timely(orbium, obstacle, N=100, run_time = 1, seed=s, cluster=True)

    s += 1

    if i % 5 == 0:  # Every five turns
        s = 0  # reset seed
        g += 0.5  # next gradient



run_simulations(iter)
