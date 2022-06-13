# !/usr/bin/env python3
"""Evolving multiple orbium in the same arena"""

# PREPARATIONS #
from lenia_package import *  # load lenia package

# FUNCTIONS #
def population(n=2):
    """return list of n number of orbium in random orientations
    and grid space"""
    orbiums = [Creature("orbium",
                        cx = np.random.randint(64),
                        cy = np.random.randint(64),
                        dir = np.random.randint(4)) for i in range(n)]
    return orbiums

##### ATTEMPT 1 #####
"""VERSION 1: EVOLVING MULTIPLE ORBIUM IN SAME ARENA
-------------------------------------------------------------
No special updating. 
"""

orbium = Creature("orbium", n=2) # evolve two orbium
obstacle = ObstacleChannel()

optimise_timely(orbium, obstacle, N=100, run_time=10)




