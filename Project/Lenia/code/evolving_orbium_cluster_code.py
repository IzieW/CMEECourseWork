# !/usr/bin/env python3

"""OPTIMISATION CODE TO SEND TO CLUSTER"""

from lenia_package import *  # Load package
import os


# Load creatures

iter = int(os.environ.get("PBS_ARRAY_INDEX"))

def run_simulations(i):

    all_gradients = np.repeat(np.arange(0.5, 10.5, 0.5), 3)  # 20 gradients

    s = i
    g = all_gradients[i]

    orbium = Creature("orbium", cluster=True)  # Initiate creature
    obstacle = ObstacleChannel(n=1, r=60, gradient=g)

    optimise_timely(orbium, obstacle, N=100, run_time=600, seed=s, cluster=True)  # Run for 10 hours each


run_simulations(iter)
