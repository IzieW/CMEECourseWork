# !/usr/bin/env python3
"""As taken from Bert Chan's google notebook, this script
gives basic code for creation of smooth, continuous Lenia system"""

##### PREPARATION ######

## IMPORTS ##
import numpy as np
import scipy as sc
from scipy.signal import convolve2d
import matplotlib.pylab as plt
from matplotlib import animation

# Silence warnings
np.warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)  # Silence warnings


## FUNCTIONS ##

def figure_world(A, cmap="viridis"):
    """Set up basic graphics of unpopulated, unsized world"""
    global img  # make final image global
    fig = plt.figure()  # Initiate figure
    img = plt.imshow(A, cmap=cmap, interpolation="nearest", vmin=0)  # Set image
    plt.title = ("World A")
    plt.close()
    return fig


def figure_asset(K, growth, cmap="viridis", K_sum=1, bar_K=False):
    """ Return graphical representation of input kernel and growth function.
    Subplot 1: Graph of Kernel in matrix form
    Subplot 2: Cross section of Kernel around center. Y: gives values of cell in row, X: gives column number
    Subplot 3: Growth function according to values of U (Y: growth value, X: values in U)
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


##### DEFINE CONTINUOUS LENIA #####
# Define values
size = 64
T = 10  # Total time
R = 10  # Kernel Radius (neighborhood size)
np.random.seed(0)
bell = lambda x, m, s: np.exp(-((x - m) / s) ** 2 / 2)  # Gaussian function

"""Generate grid with random states between 0:1"""
A = np.random.rand(size, size)  # Generate Grid with random states from 0:1

"""Create smooth ring Kernel"""
D = np.linalg.norm(np.asarray(np.ogrid[-R:R, -R:R]) + 1) / R  # Distance matrix
K = (D < 1) * bell(D, 0.5, 0.15)  # For each value in radius 1, define along gaussian curve
K = K / np.sum(K)  # Normalise


def growth(U):
    """Smooth growth function takes neighborhood sum as input.
    Defines shrinkage vs. growth along smooth bell curve of center m
    and growth width s"""
    m = 0.135
    s = 0.015
    return bell(U, m, s) * 2 - 1


def update(i):
    """Each timestep, find neighbourhoods in U,
    update each cell be 1/T growth according to neighbourhood values"""
    global A, img
    U = convolve2d(A, K, mode='same', boundary='wrap')
    A = np.clip(A + 1 / T * growth(U), 0, 1)
    img.set_array(A)
    return img,


fig = figure_world(A)
anim = animation.FuncAnimation(fig, update, frames=200, interval=20)
anim.save("results/Lenia_basic.gif", writer="imagemagick")
