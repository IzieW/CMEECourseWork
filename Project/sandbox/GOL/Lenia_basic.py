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


fig = figure_world(A, cmap="binary")  # Initiate figure world

anim = animation.FuncAnimation(fig, update, frames=50, interval=100)
anim.save("results/gol_test_new.gif", writer="imagemagick")

## 2. CONVOLUTION WITH KERNEL
"""Extend original game of life by generalising the system.
Use convolution  and a growth function. SYSTEM PATTERNS ARE UNAFFECTED AT THIS STAGE"""

size = 64
np.random.seed(0)
A = np.random.randint(2, size=(size, size))  # Initiate grid
K = np.ones((3, 3))
K[1, 1] = 0


def update_ext(i):
    global A
    """Use convolution instead of np.roll sum"""
    U = convolve2d(A, K, mode="same", boundary="wrap")
    A = (A & (U == 2)) | (U == 3)
    img.set_array(A)
    return img,


fig = figure_world(A, cmap="binary")
anim = animation.FuncAnimation(fig, update_ext, frames=50, interval=100)
anim.save("results/gen_gol.gif", writer="imagemagick")

## 3. INCREMENTAL UPDATE WITH GROWTH
"""Use incremental update by growth function instead of a conditional update. 
Growth function consists of a growth part and shrink part, 
controlled by growth and shrink ranges of the neighborhood sum"""


def figure_asset(K, growth, cmap="viridis", K_sum=1, bar_K=False):
    """Configures Graphical representations of input Kernel and growth function.
        The first plot on ax[0] demonstrates values of the Kernel across 0, 1, 2 columns
        ax[1] Gives cross section of the Kernel, ie. plots the values of row 1 (middle row of 3x3 kernel), around the target cell
        ax[2] Gives effect of Growth Kernel for different values of U. Negative or positive growth.
    """
    global R
    K_size = K.shape[0];
    K_mid = K_size // 2  # Get size and middle of Kernel
    fig, ax = plt.subplots(1, 3, figsize=(14, 2),
                           gridspec_kw={"width_ratios": [1, 1, 2]})  # Initiate figures with subplots

    ax[0].imshow(K, cmap=cmap, interpolation="nearest", vmin=0)
    ax[0].title.set_text("Kernel_K")

    if bar_K:
        ax[1].bar(range(K_size), K[K_mid, :], width=1)  # make bar plot
    else:
        ax[1].plot(range(K_size), K[K_mid, :])  # otherwise, plot normally
    ax[1].title.set_text("K cross-section")
    ax[1].set_xlim([K_mid - R - 3, K_mid + R + 3])

    if K_sum <= 1:
        x = np.linspace(0, K_sum, 1000)
        ax[2].plot(x, growth(x))
    else:
        x = np.arange(K_sum + 1)
        ax[2].step(x, growth(x))
    ax[2].axhline(y=0, color="grey", linestyle="dotted")
    ax[2].title.set_text("Growth G")
    return fig


# Reset values- same as above
size = 64
np.random.seed(0)
A = np.random.randint(2, size=(size, size))  # Initiate grid
K = np.ones((3, 3))
K[1, 1] = 0

K_sum = np.sum(K)


def growth(U):
    """Define growth function with growth/shrink ranges.
    Take neighborhood sum as input.
    The two logical values below are mutually exlusive. One will evaluate to 1
    and the other to zero.
    Returned values for each cell will be -1 or 1.
    Once clipped this gives us 1, or 0."""
    return 0 + (U == 3) - ((U < 2) | (U > 3))


def update_growth(i):
    global A
    U = convolve2d(A, K, mode='same', boundary="wrap")
    """Use incremental update and clipping. Instead of simple conditional"""  #
    # A = (A & (U==2)) | (U==3)
    A = np.clip(A + growth(U), 0, 1)  # Add growth distribution to grid and clip
    img.set_array(A)
    return img,


# Generate Kernel and Growth assets
figure_asset(K, growth, K_sum=int(K_sum), bar_K=True)

# Run game using growth function
fig = figure_world(A, cmap="binary")
anim = animation.FuncAnimation(fig, update_growth, frames=50, interval=50)
anim.save("results/growth_function_GOL.gif", writer="imagemagick")

"""Lenia has now been generalised. We can extend it to continuous cases:
1. Game of life -> larger than life (continuous space)
2. Game of life -> Primordia (continuous """
