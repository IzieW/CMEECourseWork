# !/usr/bin/env python3

"""Cluster script to test evolution of orbium with constantly decaying growth mean, and
available food sources to replenish then """

## IMPORTS ##
from distributed_equilibrium import *
import os

## FUNCTIONS ##
iter = int(os.environ.get("PBS_ARRAY_INDEX"))

def run_simulations(i):
    food_n = np.repeat([3, 6], 3)

    orbium = Creature("orbium", cluster=True)

    food = Food(n=food_n[i])

    time = 60*12

    optimise_timely(orbium, food, N=100, seed=i, run_time=time, cluster=True, name=
                    "dist_equ_m_N100_seed"+str(i)+"food"+str(food_n[i])+"t"+str(time))




run_simulations(iter)
