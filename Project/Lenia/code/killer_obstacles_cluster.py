# !/usr/bin/env python3

"""Cluster script to test evolution of orbium in environment of "killer obstalces".
Ie. every contact with obstacle causes injury to orbium. Test across several injury thresholds"""



## IMPORTS ##
from lenia_package import Creature
from killer_obstacles import *
import os

## FUNCTIONS ##
iter = int(os.environ.get("PBS_ARRAY_INDEX"))

def run_simulations(i):
    thresholds = np.repeat([10, 30],2)

    obstacles = KillerObstacle(n=3, r = 8)

    orbium = Creature("orbium", cluster=True, injury_threshold=thresholds[i])

    time = 1

    optimise_timely(orbium, [obstacles], N=100, seed=i, run_time=time, cluster=True, name=
                    "killer_obstacles_N100_seed"+str(i)+"injury"+str(thresholds[i])+"t"+str(time)+".csv")




run_simulations(iter)
