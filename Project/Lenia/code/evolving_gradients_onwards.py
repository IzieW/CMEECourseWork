# !/usr/bin/env python3
"""Continued work from evolution and runs over 10 gradient environments.
Environments where creatures last for more than a few seconds seem to favour a meshy-gridlike
layout state.

Script is dedicated to exploring environmental configurations that could potentially help
preserve our swimmers. Exploration asks the fundamental question- Which types of evolutionary environments
and selection pressures favour a swimmer over a meshy grid?"""

# Preparation
from lenia_package import *

### 1. Very dispersed gradients with less intense negative growth
"""Trial intense gradients (ie. the not very diffuse gradients which were too harsh for creatures
to even live in) with a lower all around negative growth peak"""

## EXPLORING THE RIGHT COMBINATION OF PEAKS AND GRADIENTS

"""times = []
orbium = Creature("orbium")
p = 1
for i in range(10): # gradient 1
    o = ObstacleChannel(n=1, r=60, gradient=1, peak = p)
    p = p/2  # half each time
    times.append(tuple(get_survival_time(orbium, o, summary=True)))"""
## Until peak is <0.1, not much better odds of survival

orbium = Creature("orbium")
obstacle = ObstacleChannel(n=1, r=60, gradient=1, peak = 0.015)

optimise(orbium, obstacle, fixation = 15, N=100)
