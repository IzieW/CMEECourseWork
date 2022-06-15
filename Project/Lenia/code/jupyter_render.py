# !/usr/bin/env python3
"""Script with function to load in files for rendering in jypter notebook"""
from lenia_package import *

def render_ob(orbium, obstacle, seed=4):
    orbium.initiate
    obstacle.initiate(seed=seed)
    orbium.enviro = obstacle.grid
    fig = Creature.figure_world(sum([orbium.A, orbium.enviro]))
    IPython.display.HTML(animation.FuncAnimation(fig, orbium.update_obstacle, frames=200, interval=20).to_jshtml())
