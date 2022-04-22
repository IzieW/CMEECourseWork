# !/usr/bin/env python3

"""Script creates basic Lenia system from game of life.
Taken from Bert Chan's google notebook.

    Parameters
    ------------
    R = Radius of neighbourhood
    U = Neighbourhood Sum
    A = Array/ "world"

"""

## IMPORTS ##
import numpy as np
import scipy as sc
from scipy.signal import convolve2d
import matplotlib.pylab as plt
from matplotlib import animation

## PREPARATIONS ##
np.warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)  # Silence warnings
R = 1  # Set defaults

## FUNCTIONS ##

# 1. ORIGINAL GAME OF LIFE
"""Firstly, complete Bert Chan's reformulation of Game of life"""


def figure_world(A, cmap="viridis"):
    """Set up basic graphics of unpopulated, unsized world"""
    global img  # make final image global
    fig = plt.figure()  # Initiate figure
    img = plt.imshow(A, cmap=cmap, interpolation="nearest", vmin=0)  # Set image
    plt.title = ("World A")
    plt.close()
    return fig


"""Perform Conway's game of life.
    Return saved animation of the game. NOTE TO SELF:
    Unable to open animation from within pycharm due to errors thrown
    ... only able save it as a file."""
size = 64
np.random.seed(0)
A = np.random.randint(2, size=(size, size))  # Populate 60x60 grid with random 1's and 0's


def update(i):
    """Update board according to local rules"""
    global A  # Make board global property
    """Neighborhood sum: 
            Returns array where each cell gives value of neighbourhood sum."""
    n = (-1, 0, +1)  # Define neighbourhood
    """Returns array where each cell gives value of neighbourhood sum"""
    U = sum(np.roll(A, (i, j), axis=(0, 1))
            for i in n for j in n
            if (i != 0 or j != 0))
    """Conditional Update: 
            For each cell in A, cell is true if A and U==2, or if U==3"""
    A = (A & (U == 2)) | (U == 3)
    img.set_data(A)  # Grid space img with array
    return img,

    A_blank = np.zeros_like(A)


fig = figure_world(A, cmap="binary")  # Initiate figure world

anim = animation.FuncAnimation(fig, update, frames=50, interval=100)
anim.save("gol_test_new.gif", writer="imagemagick")

## 2. CONVOLUTION WITH KERNEL
"""Extend original game of life by generalising the system.
Use convolution  and a growth function. SYSTEM PATTERNS ARE UNAFFECTED AT THIS STAGE"""

size = 64
np.random.seed(0)
A = np.random.randint(2, size=(size, size))  # Initiate grid
K = np.ones((3, 3))
K[1, 1] == 0


def update_ext(i):
    global A
    """Use convolution instead of np.roll sum"""
    U = convolve2d(A, K, mode="same", boundary="wrap")
    A = (A & (U == 2)) | (U == 3)
    img.set_array(A)
    return img,


fig = figure_world(A, cmap="binary")
anim = animation.FuncAnimation(fig, update_ext, frames=50, interval=100)
anim.save("gen_gol.gif", writer="imagemagick")

## 3. INCREMENTAL UPDATE WITH GROWTH
"""Use incremental update by growth function instead of a conditional update. 
Growth function consists of a growth part and shrink part, 
controlled by growth and shrink ranges of the neighborhood sum"""
