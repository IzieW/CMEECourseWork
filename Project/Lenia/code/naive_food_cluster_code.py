# !/usr/bin/env python3

"""Cluster script to test evolution of orbium across different environments."""

## IMPORTS ##
from lenia_package import *  # load package
from naive_food import *
import os

## FUNCTIONS ##
iter = int(os.environ.get("PBS_ARRAY_INDEX"))

def run_simulations(i):
    hostility= np.arange(0.5, 0.8, 0.05)
    orbium = Creature("orbium", cluster=True)

    if i < 7:
        food = Food(n=3, r=8)
        hostile_enviro = Atmosphere(hostility=hostility[i])
        name = "naive_food_s"+str(i)+"_N"+str(100)+"_foods_"+str(3)+"hostility"+str(hostility[i])
        optimise_layered(orbium, [food, hostile_enviro], N=1000, run_time=480, seed=i, cluster=True, name=name)
    else:
        food = Food(n=6, r=8)
        hostile_enviro = Atmosphere(hostility=hostility[i-7])
        name = "naive_food_s"+str(i)+"_N"+str(100)+"_foods_"+str(6)+"hostility"+str(hostility[i-7])
        optimise_layered(orbium, [food, hostile_enviro], N=1000, run_time=480, seed=i, cluster=True, name=name)



run_simulations(iter)

