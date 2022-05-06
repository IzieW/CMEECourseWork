# !/usr/bin/env python3

"""Script for modelling Lenia obstacles using flowers method of
layered channels connected by growth functions and kernels.
Below script outlines my own method for creating simple solid obstacles
and a gradient of negative values."""

## PREPARATIONS ##
## IMPORTS ##
import numpy as np
import scipy as sc
from scipy.signal import convolve2d
import matplotlib.pylab as plt
from matplotlib import animation

# Silence warnings
np.warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)  # Silence warnings

"""Some preliminary functions used to configure Lenia world
graphically and display kernels/growth functions used. These functions
were taken directly from Bert Chan's google notebook"""


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


############ MODELLING OBSTACLES ################
"""Below method, adapted from Flowers 2022, involves layerng two channels. 
The learning channel, A, contains our classic Lenia life form and is subject to 
traditional rules of growth according to neighbouring cells etc. 

Our obstacle channel, O, is a separate channel grid used to model our obstacles. 

The obstacle channels impact learning channels assymetrically via a Kernel (in some cases) and growth function. 
The kernel determines the "reach" of the obstacles- ie. how a life form is impacted by the obstacle. In the
case of solid obstacles, this "reach" is limited to direct overlap of cells. In these cases a kernel becomes redunant. 
However, if we wish to give obstacles a gradient of influence, a kernel can be used. 

The obstacle growth function triggers severe negative growth in all areas that have been impacted by the obstacle. """

## INITIALISE LEARNING CHANNEL ##
"""For the sake of demonstrating obstacles, all functions in this script will use the same learning channel, 
which will contain a simple swimming orbium. 

Code below to load in the learning channel is taken from Bert Chan's google notebook"""

## LOAD ORBIUM
pattern = {}
pattern["orbium"] = {"name": "Orbium", "R": 13, "T": 10, "m": 0.15, "s": 0.015, "b": [1],
                     "cells": [[0, 0, 0, 0, 0, 0, 0.1, 0.14, 0.1, 0, 0, 0.03, 0.03, 0, 0, 0.3, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0.08, 0.24, 0.3, 0.3, 0.18, 0.14, 0.15, 0.16, 0.15, 0.09, 0.2, 0, 0, 0,
                                0],
                               [0, 0, 0, 0, 0, 0.15, 0.34, 0.44, 0.46, 0.38, 0.18, 0.14, 0.11, 0.13, 0.19, 0.18, 0.45,
                                0, 0, 0],
                               [0, 0, 0, 0, 0.06, 0.13, 0.39, 0.5, 0.5, 0.37, 0.06, 0, 0, 0, 0.02, 0.16, 0.68, 0, 0, 0],
                               [0, 0, 0, 0.11, 0.17, 0.17, 0.33, 0.4, 0.38, 0.28, 0.14, 0, 0, 0, 0, 0, 0.18, 0.42, 0,
                                0],
                               [0, 0, 0.09, 0.18, 0.13, 0.06, 0.08, 0.26, 0.32, 0.32, 0.27, 0, 0, 0, 0, 0, 0, 0.82, 0,
                                0],
                               [0.27, 0, 0.16, 0.12, 0, 0, 0, 0.25, 0.38, 0.44, 0.45, 0.34, 0, 0, 0, 0, 0, 0.22, 0.17,
                                0],
                               [0, 0.07, 0.2, 0.02, 0, 0, 0, 0.31, 0.48, 0.57, 0.6, 0.57, 0, 0, 0, 0, 0, 0, 0.49, 0],
                               [0, 0.59, 0.19, 0, 0, 0, 0, 0.2, 0.57, 0.69, 0.76, 0.76, 0.49, 0, 0, 0, 0, 0, 0.36, 0],
                               [0, 0.58, 0.19, 0, 0, 0, 0, 0, 0.67, 0.83, 0.9, 0.92, 0.87, 0.12, 0, 0, 0, 0, 0.22,
                                0.07], [0, 0, 0.46, 0, 0, 0, 0, 0, 0.7, 0.93, 1, 1, 1, 0.61, 0, 0, 0, 0, 0.18, 0.11],
                               [0, 0, 0.82, 0, 0, 0, 0, 0, 0.47, 1, 1, 0.98, 1, 0.96, 0.27, 0, 0, 0, 0.19, 0.1],
                               [0, 0, 0.46, 0, 0, 0, 0, 0, 0.25, 1, 1, 0.84, 0.92, 0.97, 0.54, 0.14, 0.04, 0.1, 0.21,
                                0.05],
                               [0, 0, 0, 0.4, 0, 0, 0, 0, 0.09, 0.8, 1, 0.82, 0.8, 0.85, 0.63, 0.31, 0.18, 0.19, 0.2,
                                0.01],
                               [0, 0, 0, 0.36, 0.1, 0, 0, 0, 0.05, 0.54, 0.86, 0.79, 0.74, 0.72, 0.6, 0.39, 0.28, 0.24,
                                0.13, 0],
                               [0, 0, 0, 0.01, 0.3, 0.07, 0, 0, 0.08, 0.36, 0.64, 0.7, 0.64, 0.6, 0.51, 0.39, 0.29,
                                0.19, 0.04, 0],
                               [0, 0, 0, 0, 0.1, 0.24, 0.14, 0.1, 0.15, 0.29, 0.45, 0.53, 0.52, 0.46, 0.4, 0.31, 0.21,
                                0.08, 0, 0],
                               [0, 0, 0, 0, 0, 0.08, 0.21, 0.21, 0.22, 0.29, 0.36, 0.39, 0.37, 0.33, 0.26, 0.18, 0.09,
                                0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0.03, 0.13, 0.19, 0.22, 0.24, 0.24, 0.23, 0.18, 0.13, 0.05, 0, 0, 0,
                                0], [0, 0, 0, 0, 0, 0, 0, 0, 0.02, 0.06, 0.08, 0.09, 0.07, 0.05, 0.01, 0, 0, 0, 0, 0]]
                     }
## SET UP ##
bell = lambda x, m, s: np.exp(-((x - m) / s) ** 2 / 2)  # Gaussian function
size = 64;
mid = size // 2;
scale = 0.75;
cx, cy = 20, 20

globals().update(pattern["orbium"])  # load orbium pattern
C = np.asarray(cells)
C = sc.ndimage.zoom(C, scale, order=0)
R *= scale

"""Load learning channel"""
A = np.zeros([size, size])  # Initialise learning channel, A
A[cx:cx + C.shape[0], cy:cy + C.shape[1]] = C  # Load initial configurations into learning channel)

"""Load learning Kernel. Kernel determines each cells "neighborhood" or area of impact. 
Ie. Which neighbouring cells impact the cell in question and by how much. Orbium uses a single-ring
kernel, which attributes most impact to cells a given distance away in all directions."""
D = np.linalg.norm(np.asarray(np.ogrid[-mid:mid, -mid:mid]) + 1) / R  # create distance matrix
K = (D < 1) * bell(D, 0.5, 0.15)  # Transform all distances within radius 1 along smooth gaussian gradient
K = K / np.sum(K)  # Normalise between 0:1
# Fourier transformations used as a more efficient method than convolution
fK = np.fft.fft2(np.fft.fftshift(K)) # Pre-Fourier transform K


def learning_growth(U):
    """Growth function determines how learning channels are impacted by the cells in their neighbourhood.
    Orbium uses a smooth gaussian growth function"""
    return bell(U, m, s) * 2 - 1


### SOLID OBSTACLES ####
""" Below creates obstacle channel containing a solid obstacle, which only impacts the 
creature on direct contacts (overlap of cells)."""

# Create solid obstacle channel
O = np.zeros([size, size])  # Grid same size as Learning channel
O[35:45, 35:45] = 1  # Initialise arbitrary area populated by obstacle

"""Create solid-obstacle kernel"""
sigmoid = lambda x: 1 / (1 + np.exp(-x))
obstacle_k = lambda x: np.exp(-((x / 2) ** 2) / 2) * sigmoid(-10 * (x / 2 - 1))
KO = (D < 0.05) * obstacle_k(D)  # Only impacts when direclty on obstacle
fKO = np.fft.fft2(np.fft.fftshift(KO))  # Fourier transform

As = [A, O]  ## Create list of channels


def obstacle_growth(U):
    """Determines how learning channel cells are impacted by
    obstacle cells. Below gives function for severe negative growth for any cell values > 0.
    For all values greater than 0, growth function causes ten times negative growth"""
    return -10 * np.maximum(0, (U - 0.001))


# plt.matshow(sum(As))  # Show state of grid at t=0

def update(i):
    """Function determines how grid is updated with each
    interval. Each interval t, each cell's specified neighbourhood
    is assessed and the cell either grows or decays according to
    its local rules. In the case of multiple channels,
    the learning channel is updated according to a sum of growth
    caused by its domestic cells and obstacle channel."""
    global As, img  # As = [A, O]
    U1 = convolve2d(As[0], K, mode="same", boundary ="wrap")
    #U1 = np.real(np.fft.ifft2(fK*np.fft.fft2(A)))
    U2 = convolve2d(As[1], KO, mode = "same", boundary="wrap")
    #U2 = np.real(np.fft.ifft2(fKO * np.fft.fft2(O)))
    # Update learning channel by summing growth in both channels
    As[0] = np.clip(As[0] + (1 / T)* (learning_growth(U1) + obstacle_growth(U2)), 0, 1)
    img.set_array(sum(As))  # Render visuals by summing resultant contents of both channels
    return img,


#### RENDER SIMULATION ###
As = [A, O]
np.random.seed(0)  # set seed
fig = figure_world(sum(As))
anim = animation.FuncAnimation(fig, update, frames=200, interval=20)
anim.save("results/solid_obstacles_2.gif", writer="imagemagick")

### MODELLING OBSTACLE GRADIENTS ###
"""Rather than solid obstacle, below models a gradient of 
negative values starting from the top of the board to the end"""

def gradient_update(i):
    global As, img
    U1 = convolve2d(As[0], K, mode="same", boundary="wrap")
    # U2 - no need for convolution by kernel
    # Channel itself contains all values
    # A = np.clip(A + 1/T*(growth(U1)), 0, 1)
    As[0] = np.clip(As[0] + 1 / T * (growth(U1) + obstacle_growth(grad_A)), 0, 1)
    img.set_array(sum(As))
    return img,

# ATTEMPT ONE: Gradient using simply count
    # Environment far too hostile- Orbium cannot even manifest
def normal_count_grad():
    """Return gradient channel A using normal count"""
    grad_A = np.zeros([size, size])
    for i in range(size): grad_A[i,] = i
    grad_A = np.flip(grad_A / size)  # Normalise and flip so highest values are on top
    return grad_A

## ATTEMPT TWO: Exponential decay
    # Steaper decline of hostile values. Much of board remains safe.

def exponential_gradient(gradient_stretch):
    """Populate channel using exponential decay.
    Gradient stretch gives value exp curve is stretched by"""
    x = np.exp(-np.arange(size)/gradient_stretch)
    g = np.zeros([size, size])
    for i in range(size):
        g[i,] = x[i]
    return g

gradient_channel = exponential_gradient(1) # simple exponential decay not diffuse enough
# Values too similar to solid obstacle

gradient_channel = exponential_gradient(50)  # Too diffuse, entire board is safe


# Need to find correct balance.
    # Can edit gradient channel or negative growth induced by obstacle channel
