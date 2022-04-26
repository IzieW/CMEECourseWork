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
plt.savefig("results/basic_growth_func.png")

# Run game using growth function
fig = figure_world(A, cmap="binary")
anim = animation.FuncAnimation(fig, update_growth, frames=50, interval=50)
anim.save("results/growth_function_GOL.gif", writer="imagemagick")

"""System has now been generalised. We can extend it to continuous cases:
1. Game of life -> larger than life (continuous space)
2. Game of life -> Primordia (continuous states)
3. Primordia -> Lenia (continuous states-space-time"""

## 1. LARGER THAN LIFE
"""In this CA, the convolutional Kernel is englarged to radius R, (extending the 
Moore Neighbourhood). 
Uses the periodic pattern Bosco- a particular type of oscillator"""
# Load Bosco- sample creates
pattern = {}  # Dictionary of creatures
pattern["bosco"] = {"name": "Bosco", "R": 5, "b1": 34, "b2": 45, "s1": 34, "s2": 58,
                    "cells": [[0, 0, 0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
                              [0, 1, 1, 0, 0, 1, 1, 1, 1, 0], [1, 1, 0, 0, 0, 1, 1, 1, 1, 1],
                              [1, 0, 0, 0, 0, 0, 1, 1, 1, 1], [1, 1, 0, 0, 0, 0, 1, 1, 1, 1],
                              [1, 1, 1, 0, 1, 1, 1, 1, 1, 1], [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                              [0, 1, 1, 1, 1, 1, 1, 1, 0, 0], [0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
                              [0, 0, 0, 1, 1, 1, 0, 0, 0, 0]]
                    }

# Set values
size = 64;
cx, cy = 10, 10
"""Instead of randint population, load bosco pattern, including
parameters R, b1, b2, s1, s2, and cells"""
globals().update(pattern["bosco"])
C = np.asarray(cells)
A = np.zeros([size, size])
A[cx:cx + C.shape[0], cy:cy + C.shape[1]] = C
"""Extend neighborhood (includes self)"""
K = np.ones((2 * R + 1, 2 * R + 1))
K_sum = np.sum(K)


def growth(U):
    """Bosco's rule:
     b1..b2 is birth range
     s1..s2 is stable range (outside of s1..s2 is shrink).
     Below, if U is within the stable range, the second term will evaluate to zero.
     If it is outside of the stable range, it will evalate to 1.
     Needs to be within stable and birth range to be born"""
    return 0 + ((U >= b1) & (U <= b2)) - ((U < s1) | (U > s2))


def update(i):
    global A
    U = convolve2d(A, K, mode="same", boundary="wrap")
    A = np.clip(A + growth(U), 0, 1)
    img.set_array(A)
    return img,


figure_asset(K, growth, K_sum=K_sum, bar_K=True)
plt.savefig("results/larger_than_life.png")

fig = figure_world(A, cmap="binary")
anim = animation.FuncAnimation(fig, update, frames=200, interval=50)
anim.save("results/larger_than_life.gif", writer="imagemagick")

## 2. PRIMORDIA
"""Create continuous states, time and space in game of life.

Firstly, allow a gradient of states between 0 and 1. Everything (growth/shrink) becomes
scaled up by number of states. The resulting patterns are complex, and the precurser to Lenia"""

# set scales
size = 64
states = 12  # multiple states
np.random.seed(0)
A = np.random.randint(states + 1, size=(size, size))  # Generate with random states
K = np.ones((3, 3))
K[1, 1] = 0
K_sum = states * np.sum(K)


def growth(U):
    """Scale everything up by number of states"""
    return 0 + ((U >= 20) & (U <= 24)) - ((U <= 18) | (U >= 32))

def update(i):
    global A
    U = convolve2d(A, K, mode="same", boundary="wrap")
    A = np.clip(A + growth(U), 0, states)
    """Normalise pixels for image"""
    img.set_array(A / states)
    return img,


figure_asset(K, growth, K_sum=K_sum, bar_K=True)
plt.savefig("results/primordia_growth.png")

fig = figure_world(A / states)
anim = animation.FuncAnimation(fig, update, frames=200, interval=20)
anim.save("results/Primorida.gif", writer="imagemagick")


## NORMALIZED KERNEL
"""Can normalise states, kernel and growth to make further generalisations easier.
While states are integers, everytime we change the number of states, we will need to change
all other properties as well.

By restricting states between 0 and 1, we are able to make generalisations easier. 

The resulting patterns will not change qualitatively"""

# Normalise Kernel, and growth/shrink ranges

size = 64
states = 12
np.random.seed(0)
A = np.random.randint(states+1, size=(size, size))
K = np.ones((3,3))
K[1,1] = 0
K_sum = states * np.sum(K)

"""normalise kernel"""
K = K/K_sum

def growth(U):
    """Normalise growth/shrink ranges, divide each number by K_sum"""
    return 0 + ((U >= 0.2) & (U <= .25)) - ((U <= 0.18) | (U >= 0.33))

def update(i):
    global A
    U = convolve2d(A, K, mode="same", boundary="wrap")
    A = np.clip(A + growth(U), 0, states)
    img.set_array(A / states)
    return img,

figure_asset(K, growth, bar_K=True)
plt.savefig("results/normalised_primordial.png")

fig = figure_world(A/states)
anim = animation.FuncAnimation(fig, update, frames= 200, interval = 20)
anim.save("results/normalised_kernel_primoridal.gif", writer = "imagemagick")


## CONTINUOUS STATES AND TIME
"""Next, we normalise states from discrete numbers to a range [0.0, 1.0]

States become continous (effectively infinite, but technically still subject to precision
of floating point numbers. Number of states is no longer useful.

Will also define update frequency (T), and scale down incremental updates by a factor of dt, 1/T.
By taking T to infinity, ie. very small timesteps, time will become continuous. 

The resulting patterns do not change qualitatively."""

T = 10
np.random.seed(0)
A = np.random.rand(size, size)  # random states [0:1]
K = np.ones((3,3))
K[1,1] = 0
K = K/np.sum(K)  # normalise

def growth(U):
    """Normalise growth/shrink ranges, divide each number by K_sum"""
    return 0 + ((U >= 0.2) & (U <= .25)) - ((U <= 0.19) | (U >= 0.33))

def update(i):
    global A
    U = convolve2d(A, K, mode="same", boundary="wrap")
    A = np.clip(A + 1/T*growth(U), 0, 1)
    img.set_array(A)
    return img,

figure_asset(K, growth, bar_K=True)

fig = figure_world(A)
anim = animation.FuncAnimation(fig, update, frames=200, interval = 20)
anim.save("results/continuous_time_space_primordial.gif", writer= "imagemagick")

## LENIA
"""Adding to the above, space can be made continuous as well. 

Define Kernel radius R. The convolution kernel is englarged to readius R, but still rectangular in shape. 
As R approaches infinity, space becomes continuous. 

The resulting pattern acquires a sort of fluid-like quality"""

R = 5
np.random.seed(0)
A = np.random.rand(size, size)  # Populate grid with random states
"""Larger rectanglar kernel"""
K = np.ones((2 *R+1, 2*R+1)); K[R, R] = 0  # Center is still zero
K = K/np.sum(K)  # normalise

def growth(U):
    """Normalise growth/shrink ranges, divide each number by K_sum"""
    return 0 + ((U >= 0.12) & (U <= .15)) - ((U <= 0.12) | (U >= 0.15))

def update(i):
    global A
    U = convolve2d(A, K, mode="same", boundary="wrap")
    A = np.clip(A + 1/T*growth(U), 0, 1)
    img.set_array(A)
    return img,

figure_asset(K, growth, bar_K=True)
plt.savefig("results/Lenia_growth_basic.png")

fig = figure_world(A)
anim = animation.FuncAnimation(fig, update, frames=200, interval = 20)
anim.save("results/Lenia_basic.gif", writer="imagemagick")


## RING KERNEL
"""Hand draw a ring-like Kernel with the same radius. 

The circular shape of the Kernel removes orthogonal bias (ie. the horizontal and vertical stripes) 
in the patterns, changing them from anisotropic (dependent on direction) to isotrophic (equal in all directions).
THe ring-like structure also resembles smoothlife"""

np.random.seed(0)
A = np.random.rand(size, size)

# Create ring-like Kernel
K = np.asarray([
[0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
[0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
[1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1],
[1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1],
[1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1],
[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
[0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
[0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0]])
K = K / np.sum(K)

def growth(U):
  return 0 + ((U>=0.12)&(U<=0.15)) - ((U<0.12)|(U>0.15))

figure_asset(K, growth, bar_K=True)

fig = figure_world(A)
anim = animation.FuncAnimation(fig, update, frames=200, interval = 20)
anim.save("results/ring_kernel_Lenia.gif", writer="imagemagick")


## SMOOTH KERNEL
"""Can also generate the kernel using a smooth bell-shaped function. (Gaussian)

First, define a distance array from the center, normalised by factor 1/R, then feed it into 
the bell shaped function. The result is a smooth ring-shaped Kernel.

By convolving this Kernel, array U becomes the weighted neighbor sum with 
the kernel being the weights. The patterns became smoother (though hardly observable given the small
world"""
# Define gaussian function. For each number produces point in curve.
bell = lambda x, m, s: np.exp(-((x-m)/s)**2/2)

T = 10
size=64
R = 10
np.random.seed(0)
A = np.random.rand(size, size)

"""Smooth ring-like kernel: 
Define distance array D. Function below produces a Euclidean distance matrix.
Square root of the sum of abolsute values squared"""

D = np.linalg.norm(np.asarray(np.ogrid[-R:R, -R:R]) + 1)/R
K = (D<1) * bell(D, 0.5, 0.15) ## All distances within radius 1, transformed along gaussian gradient
K = K/np.sum(K)  # Normalise

def growth(U):
    return 0 + ((U>=0.12)&(U<=0.15)) - ((U<0.12)|(U>0.15))

def update(i):
    global A, img
    U = convolve2d(A, K, mode="same", boundary="wrap")
    A = np.clip(A + 1/T * growth(U), 0, 1)
    img.set_array(A)
    return img,

figure_asset(K, growth)
plt.savefig("results/smooth_kernel.png")
fig = figure_world(A)
anim = animation.FuncAnimation(fig, update, frames=200, interval=20)
anim.save("results/smooth_kernel.gif", writer="imagemagick")

## SMOOTH GROWTH
"""The growth function can also be replaced by smooth bell-shaped function, 
with parameters growth center (m) and growth width (s). This makes the patterns even smoother

With everything smoothed, we finally arrive at Lenia (meaning "smooth" in Latin)"""

bell = lambda x, m, s: np.exp(-((x-m)/s)**2 / 2)
size = 64
T = 10
R = 10
np.random.seed(0)
A = np.random.rand(size, size)
D = np.linalg.norm(np.asarray(np.ogrid[-R:R, -R:R]) + 1) / R
K = (D<1) * bell(D, 0.5, 0.15)
K = K / np.sum(K)

def growth(U):
    """Smooth growth function"""
    #  return 0 + ((U>=0.12)&(U<=0.15)) - ((U<0.12)|(U>0.15))
    m = 0.135
    s = 0.015
    return bell(U, m, s)*2-1

def update(i):
    global A, img
    U = convolve2d(A, K, mode='same', boundary='wrap')
    A = np.clip(A + 1/T * growth(U), 0, 1)
    img.set_array(A)
    return img,

figure_asset(K, growth)
fig = figure_world(A)
anim = animation.FuncAnimation(fig, update, frames=200, interval=20)
anim.save("results/smooth_lenia.gif", writer="imagemagick")
