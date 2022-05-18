# !/usr/bin/env python3

"""OPTIMISATION CODE TO SEND TO CLUSTER"""

from lenia_package import *  # Load package

# Load creatures
orbium = Creature("orbium")
obstacle = ObstacleChannel(n=1, r=60, gradient=10)

optimise_timely(orbium, obstacle, N = 100, runtime= 20)

