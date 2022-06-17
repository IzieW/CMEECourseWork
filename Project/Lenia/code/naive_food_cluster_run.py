# !/usr/bin/env python3

"""Cluster script to test evolution of orbium across different environments."""

## IMPORTS ##
from lenia_package import *  # load package
from developing_food import *
import os

## FUNCTIONS ##
iter = int(os.environ.get("PBS_ARRAY_INDEX"))

def run_simulations(i):
    food = Food(n=3, r=8)
    obstacle = ObstacleChannel(n=5, r=8)
    hostile_enviro = Atmosphere(hostility = 0.5)

    orbium = Creature("orbium", cluster=True)






