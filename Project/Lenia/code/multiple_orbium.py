# !/usr/bin/env python3
"""Evolving multiple orbium in the same arena"""

# PREPARATIONS #
from lenia_package import *
import os

iter = int(os.environ.get("PBS_ARRAY_INDEX"))

def run_simulations(i):
    n = np.repeat([1, 2, 3, 4, 5, 10], 3)

    orbium = Creature("orbium", n=n[i], cluster=True)
    obstacle = ObstacleChannel(n=5, r=8)

    optimise_timely(orbium, obstacle, N=100, run_time=360, seed=i, cluster=True)


run_simulations(iter)
