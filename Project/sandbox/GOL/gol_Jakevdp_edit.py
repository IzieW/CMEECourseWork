# !/usr/bin/env python3
"""Minimal Model of Conway's game of life:
Game composed of two dimensional grid whose dynamics mimic chaotic patterns of growth in colony of bacteria.
Each cell consists of zero:one state "living" or "Dead"

Generation to generation the rules are as follows:
 Over population: If cell is surrounded by two or more living cells , the cell dies.
 Stasis: If a cell is surrounded by two or three living cells, it survives.
 Underpopulation: If a cell is surrounded by less than two living cells, it dies
 Reproduction: If a dead cell is surrounded by exactly three living cells, it becomes a living cell
"""

## IMPORTS ##
import numpy as np
import matplotlib.pyplot as plt
## FUNCTIONS ##
# Two possible ways of completing a time step:
# Using generator expressions and convolve2d from scipy
"""Two possible ways of completing a timestep below:
Both return a matrix where each cell is the sum of its neighbourhood.
Author specifies that these are not very performant, but can suffice
for sufficiently small examples"""


def life_step_1(x):
    """Game of life step using generator expression"""
    nbrs_count = sum(np.roll(np.roll(x, i, 0), j, 1)
                     for i in (-1, 0, 1) for j in (-1, 0, 1)
                     if (i != 0 or j != 0))
    return (nbrs_count == 3) | (x & (nbrs_count == 2))


def life_step(x):
    """Game of life using scipy tools"""
    from scipy.signal import convolve2d
    nbrs_count = convolve2d(x, np.ones((3, 3)), mode="same", boundary='wrap') - x
    return (nbrs_count == 3) | (x & (nbrs_count == 2))

"""Classically, game takes place of infinite plane, flat plane. 
These methods make use of torroidal geometry where grids 
wrap from top to bottom and left to right."""

# VISUALISE the results using matplotlib animation submodule
# JSAnimation package.

from JSAnimation.IPython_display import display_animation, anim_to_html
from matplotlib import animation


def life_animation(x, dpi=10, frames=10, interval=300, mode="loop"):
    """Produce a Game of Life Animation

    Parameters
    ------------
    X: array_like
        a two-dimensional numpy array showing the game board
    dpi: integer
        the number of dots per inch in the resulting anmation.
        This contols the size of the game board on the screen.
    frames: integer
        The number of frames to compute for the animation
    interval: float
        The time interval (in milliseconds) between frames
    mode: string
        The default mode fo the animation. Option include ['loop'|'once'| reflect]
    """

    x = np.asarray(x)  # Convert to np array
    assert x.ndim == 2  # Throw error if x is not 2 dimensions
    x = x.astype(bool)  # Convert all 0 and 1 to true/false

    x_blank = np.zeros_like(x)  # Generates matrix of zeros in same shape as x
    figsize = (x.shape[1] * 1. / dpi, x.shape[0] * 1. / dpi)

    fig = plt.figure(figsize=figsize, dpi=dpi)  # Initiate figure with given size and dots per inch (dpi)
    ax = fig.add_axes([0, 0, 1, 1], xticks=[], yticks=[], frameon=False)
    im = ax.imshow(x, cmap=plt.cm.binary, interpolation="nearest")  # Arrange features of image (colour etc)
    im.set_clim(-0.05, 1)  # make background grey

    # Initialisation function
    def init():
        """Plot the background of each frame"""
        im.set_data(x_blank)
        return im,

    def animate(i):
        """Animation function, called sequentially"""
        im.set_data(animate.x)
        animate.x = life_step(animate.x)
        return im,

    animate.x = x
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=frames, interval=interval)

    return display_animation(anim, default_mode=mode)



np.random.seed(0)
x = np.zeros((30, 40), dtype=bool)
r = np.random.random((10, 20))
x[10:20, 10:30] = (r > 0.75)
life_animation(x, dpi=10, frames=40, mode='once')

