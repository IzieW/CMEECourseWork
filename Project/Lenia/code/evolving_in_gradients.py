# !/usr/bin/env python3

"""Evolve Orbium in environment with gradient of negative values"""

## IMPORTS ###
from lenia_package import *  # Load all lenia functions
import time
orbium = Creature("orbium")  # Load orbium

"""1. Configuration 1: 
Single obstacle diffused across entire grid board"""
obstacle = ObstacleChannel(n=1, r=60, gradient=10)

start = time.time()
optimise(orbium, obstacle, N=100, fixation=15)  # Optimise in population of 100
end = time.time()
total_time = end-start
