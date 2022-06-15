# !/usr/bin/env python3

"""Evolve lenia across different population sizes.
Send to cluser"""

from lenia_package import *
import os

iter = int(os.environ.get("PBS_ARRAY_INDEX"))

def run_simulations(i):

    pop_sizes = np.repeat([10, 10**2, 10**4, 10**6, 10**8], 3)
    seeds = np.repeat([1, 2, 3], 5)
    orbium = Creature("orbium", cluster=True)
    obstacle = ObstacleChannel(n=5, r=8)

    optimise_timely(orbium, obstacle, N=pop_sizes[i], run_time=360, seed=i, cluster=True)


run_simulations(iter)

