# !/usr/bin/env python3
"""Script executing minimal version of Conway's game of life.
Code inspired by that of Jake vdp (http://jakevdp.github.io/blog/2013/08/07/conways-game-of-life/)
Ammended to not use his package JSAnimate, instead only uses animation packages within
pyplot
"""

## IMPORTS ##
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
from matplotlib import animation


## FUNCTIONS ##

def life_step(x):
    """Single time step in game of life using scipy.
    Convolves each cell with neighbourhood kernal and takes sum of kernal minus cell value.
    Updates the entire board according to local rules of birth, death and survival"""
    from scipy.signal import convolve2d
    nbrs_count = convolve2d(x, np.ones((3, 3)), mode="same",
                            boundary='wrap') - x  # Produces a matrix where each cell represents its neighbourhood sum
    return (nbrs_count == 3) | (x & (nbrs_count == 2))


def animate_life(x, dpi=10, frames=10, interval=200, mode="loop"):
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

    # Handle input x
    x = np.asarray(x)  # Convert to np array
    assert x.ndim == 2  # Throw error if x is not 2 dimensions
    x = x.astype(bool)  # Convert all 0 and 1 to true/false

    # Initiate figure
    x_blank = np.zeros_like(x)  # Create blank plot of x, for background of animation

    figsize = (x.shape[1] * 1. / dpi, x.shape[0] * 1. / dpi)
    dpi = 10
    fig = plt.figure(figsize=figsize, dpi=dpi)  # Initiate figure with given size and dots per inch (dpi)
    ax = fig.add_axes([0, 0, 1, 1], xticks=[], yticks=[], frameon=False)
    im = ax.imshow(x, cmap=plt.cm.binary, interpolation="nearest")  # Arrange features of image (colour etc)
    im.set_clim(-0.05, 1)  # make background grey

    # Set up animation
    def init():
        """Initialise blank function plot"""
        im.set_data(x_blank)
        return im,

    def animate(i):
        """Animation function. Called sequentially with animation.FunAnime below.
        Updates x with new life step and sets data in image (im)"""
        im.set_data(animate.x)
        animate.x = life_step(animate.x)
        return im,

    animate.x = x  # Set initial conditions for x

    # Animate
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=10, interval=300)
    plt.show()


## TRY THIS CONFIGURATION OF X
np.random.seed(123)
x = np.zeros((30, 40), dtype=bool)
r = np.random.random((10, 20))  # Populate middle section with lives randomely
x[10:20, 10:30] = (r > 0.75)

animate_life(x)
