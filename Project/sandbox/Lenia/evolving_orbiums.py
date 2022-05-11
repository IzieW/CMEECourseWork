# !/usr/bin/env python3

"""Exploration of evolving Orbium life forms
in a simple obstacle space according to evolutionary optimisation techniques
described in Godany, Khatri and Goldstein 2017."""

#### PREPARATION ####
## IMPORTS ##
import numpy as np
import scipy as sc
from scipy.signal import convolve2d
import matplotlib.pylab as plt
from matplotlib import animation
import csv
# Silence warnings
np.warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)  # Silence warnings
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
def figure_asset_list(Ks, nKs, growth, kernels, use_c0=False, cmap='viridis', K_sum=1):
    global R
    K_size = Ks[0].shape[0];
    K_mid = K_size // 2
    fig, ax = plt.subplots(1, 3, figsize=(14, 2), gridspec_kw={'width_ratios': [1, 2, 2]})
    if use_c0:
        K_stack = [np.clip(np.zeros(Ks[0].shape) + sum(K / 3 for k, K in zip(kernels, Ks) if k['c0'] == l), 0, 1) for l
                   in range(3)]
    else:
        K_stack = Ks[:3]
    ax[0].imshow(np.dstack(K_stack), cmap=cmap, interpolation="nearest", vmin=0)
    ax[0].title.set_text('kernels Ks')
    X_stack = [K[K_mid, :] for K in nKs]
    ax[1].plot(range(K_size), np.asarray(X_stack).T)
    ax[1].title.set_text('Ks cross-sections')
    ax[1].set_xlim([K_mid - R - 3, K_mid + R + 3])
    x = np.linspace(0, K_sum, 1000)
    G_stack = [growth(x, k['m'], k['s']) * k['h'] for k in kernels]
    ax[2].plot(x, np.asarray(G_stack).T)
    ax[2].axhline(y=0, color='grey', linestyle='dotted')
    ax[2].title.set_text('growths Gs')
    return fig

### EVOLVING ORBIUM ####
## LOAD ORBIUM ##
orbium = {"name": "Orbium", "R": 13, "T": 10, "m": 0.15, "s": 0.015, "b": [1],
          "cells": [[0, 0, 0, 0, 0, 0, 0.1, 0.14, 0.1, 0, 0, 0.03, 0.03, 0, 0, 0.3, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0.08, 0.24, 0.3, 0.3, 0.18, 0.14, 0.15, 0.16, 0.15, 0.09, 0.2, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0.15, 0.34, 0.44, 0.46, 0.38, 0.18, 0.14, 0.11, 0.13, 0.19, 0.18, 0.45, 0, 0, 0],
                    [0, 0, 0, 0, 0.06, 0.13, 0.39, 0.5, 0.5, 0.37, 0.06, 0, 0, 0, 0.02, 0.16, 0.68, 0, 0, 0],
                    [0, 0, 0, 0.11, 0.17, 0.17, 0.33, 0.4, 0.38, 0.28, 0.14, 0, 0, 0, 0, 0, 0.18, 0.42, 0, 0],
                    [0, 0, 0.09, 0.18, 0.13, 0.06, 0.08, 0.26, 0.32, 0.32, 0.27, 0, 0, 0, 0, 0, 0, 0.82, 0, 0],
                    [0.27, 0, 0.16, 0.12, 0, 0, 0, 0.25, 0.38, 0.44, 0.45, 0.34, 0, 0, 0, 0, 0, 0.22, 0.17, 0],
                    [0, 0.07, 0.2, 0.02, 0, 0, 0, 0.31, 0.48, 0.57, 0.6, 0.57, 0, 0, 0, 0, 0, 0, 0.49, 0],
                    [0, 0.59, 0.19, 0, 0, 0, 0, 0.2, 0.57, 0.69, 0.76, 0.76, 0.49, 0, 0, 0, 0, 0, 0.36, 0],
                    [0, 0.58, 0.19, 0, 0, 0, 0, 0, 0.67, 0.83, 0.9, 0.92, 0.87, 0.12, 0, 0, 0, 0, 0.22, 0.07],
                    [0, 0, 0.46, 0, 0, 0, 0, 0, 0.7, 0.93, 1, 1, 1, 0.61, 0, 0, 0, 0, 0.18, 0.11],
                    [0, 0, 0.82, 0, 0, 0, 0, 0, 0.47, 1, 1, 0.98, 1, 0.96, 0.27, 0, 0, 0, 0.19, 0.1],
                    [0, 0, 0.46, 0, 0, 0, 0, 0, 0.25, 1, 1, 0.84, 0.92, 0.97, 0.54, 0.14, 0.04, 0.1, 0.21, 0.05],
                    [0, 0, 0, 0.4, 0, 0, 0, 0, 0.09, 0.8, 1, 0.82, 0.8, 0.85, 0.63, 0.31, 0.18, 0.19, 0.2, 0.01],
                    [0, 0, 0, 0.36, 0.1, 0, 0, 0, 0.05, 0.54, 0.86, 0.79, 0.74, 0.72, 0.6, 0.39, 0.28, 0.24, 0.13, 0],
                    [0, 0, 0, 0.01, 0.3, 0.07, 0, 0, 0.08, 0.36, 0.64, 0.7, 0.64, 0.6, 0.51, 0.39, 0.29, 0.19, 0.04, 0],
                    [0, 0, 0, 0, 0.1, 0.24, 0.14, 0.1, 0.15, 0.29, 0.45, 0.53, 0.52, 0.46, 0.4, 0.31, 0.21, 0.08, 0, 0],
                    [0, 0, 0, 0, 0, 0.08, 0.21, 0.21, 0.22, 0.29, 0.36, 0.39, 0.37, 0.33, 0.26, 0.18, 0.09, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0.03, 0.13, 0.19, 0.22, 0.24, 0.24, 0.23, 0.18, 0.13, 0.05, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0.02, 0.06, 0.08, 0.09, 0.07, 0.05, 0.01, 0, 0, 0, 0, 0]]
          }
theta = [orbium[i] for i in ["R", "T", "m", "s", "b"]]  # save paramters

##### CONFIGURE ENVIRONMENT ###
## CONSTANTS ##
bell = lambda x, m, s: np.exp(-((x - m) / s) ** 2 / 2)  # Gaussian function
size = 64;
cx, cy = 20, 20
mid = size // 2;
C = np.asarray(orbium["cells"])


def learning_kernel(parameters, fourier=False):
    """Create kernel for learning channel"""
    R = parameters[0]
    D = np.linalg.norm(np.asarray(np.ogrid[-mid:mid, -mid:mid]) + 1) / R  # create distance matrix
    K = (D < 1) * bell(D, 0.5, 0.15)  ## Transform all distances within radius 1 along smooth gaussian gradient
    K = K / np.sum(K)  # Normalise between 0:1
    if fourier:
        return np.fft.fft2(np.fft.fftshift(K))  # fourier transform
    else:
        return K


#### FUNCTIONS #####
def load_learning():
    """Load learning channel"""
    A = np.zeros([size, size])  # Initialise learning channel, A
    A[cx:cx + C.shape[0], cy:cy + C.shape[1]] = C  # Load orbium initial configurations into learning channel
    return A


def load_obstacles(n, r=5, seed = 0):
    """Load obstacle channel with random configuration
    of n obstacles with radius r"""
    # Sample center point coordinates a, b
    #np.random.seed(seed)
    O = np.zeros([size, size])
    for i in range(n):
        mid_point = tuple(np.random.randint(0, size - 1, 2))
        O[mid_point[0]:mid_point[0] + r, mid_point[1]:mid_point[1] + r] = 1  # load obstacles
    return O


def learning_growth(U, parameters=1):
    """Define growth of learning channel cell according to neighbourhood values U.
    Take mean and std from life form parameters"""
    m = 0.135
    s = 0.015
    return bell(U, m, s) * 2 - 1


def obstacle_growth(U):
    """Define growth of learning channel according to overlap with obstacles"""
    return -10 * np.maximum(0, (U - 0.001))


def update(i):
    global As, img
    U1 = convolve2d(As[0], K, mode="same", boundary="wrap")
    # A = np.clip(A + 1/T*(growth(U1)), 0, 1)
    As[0] = np.clip(As[0] + 1 / T * (learning_growth(U1) + obstacle_growth(O)), 0, 1)
    img.set_array(sum(As))
    return img,

"""
# Test one round
globals().update(orbium)
O = load_obstacles(n=3)
K = learning_kernel(parameters=theta)
As = [load_learning(), load_obstacles(n=3)]
fig = figure_world(sum(As))
anim = animation.FuncAnimation(fig, update, frames=200, interval=20)
anim.save("results/test_space.gif", writer="imagemagick")"""

## Test survival based on lack of cells in A
def update_man(A, K, parameters):
    """One time step of Lenia, takes current grid A and returns one update"""
    T = parameters[1]
    U1 = np.real(np.fft.ifft2(K*np.fft.fft2(A)))
    #U1 = convolve2d(A, K, mode="same", boundary="wrap")
    # A = np.clip(A + 1/T*(growth(U1)), 0, 1)
    A = np.clip(A + 1 / T * (learning_growth(U1, parameters) + obstacle_growth(O)), 0, 1)
    return A



def get_t(A, O, parameters):
    """Run simulation with input learning channel A,
     and obstacle configuration O.

     Return time until life form dissolves"""

    K = learning_kernel(parameters, fourier=True)  # Get kernel

    status = np.sum(A)
    t = 0  # Set time
    while (status > 0) & (t < 1000):  # While there are still cells in the learning channel
        t += 1  # Record time
        #plt.matshow(A+O)
        A = update_man(A, K, parameters)  # Update channel
        status = np.sum(A)  # update sum
    return t

# One round
np.random.seed(0)
A = load_learning()
O = load_obstacles(n=3)
get_t(A, 0, parameters=theta)

def mutate(p):
    """Mutate input parameter p"""
    return np.exp(np.log(p) + np.random.uniform(low=-0.2, high=0.2))

def crude_evolve(parameters, seed=0):
    """Run one round of evolutionary optimisation.
    From input parameters, mutate one parameter chosen at random.
    Run simulations on both wild_type and mutant on same obstacle configuration.

    Calculate fitness from survival time, and return winning set of parameters."""
    A = load_learning()
    np.random.seed(seed)  # Set seed
    wild_type = parameters[:]
    mutant_type = parameters[:]
    x = np.random.randint(0, len(parameters)-1)  # Choose random index from parameter range
    print(x)
    mutant_type[x] = mutate(mutant_type[x])  # Mutate parameter in mutant type
    print(wild_type)
    print(mutant_type)

    O = load_obstacles(n=3)
    tm = get_t(A, O, mutant_type)
    tw = get_t(A, O, wild_type)

    # Calculate selection coefficient from survival time
    s = (tm - tw)/tw
    pfix = 2*s  # probability of fixation

    # Draw random n, 0 <= n <= 1
    n = np.random.sample(1)[0]
    if pfix >= n:
        print("Accept mutation")
        return mutant_type
    elif pfix < n:
        print("Reject Mutation")
        return wild_type

def run_one(parameters, seed=0):
    """Run one round of evolutionary optimisation.
    From input parameters, mutate one parameter chosen at random.
    Run simulations on 10 wild_type and mutant: Each wild-mutant couple
    experience same obstacle configuration, and configuration changes
    ten times for each group.

    Calculate overall stochastic fitness of mutant and return winning set of parameters"""
    A = load_learning()
    #np.random.seed(seed)  # Set seed
    wild_type = parameters[:]
    mutant_type = parameters[:]
    x = np.random.randint(0, len(parameters)-1)  # Choose random index from parameter range
    mutant_type[x] = mutate(mutant_type[x])  # Mutate parameter in mutant type


    # Run mutant and wild parameters over 10 obstacle configurations
    tm, tw = 0, 0  # Survival time for mutant, wild type
    for i in range(10):
        np.random.seed(i)  # Each with own seed
        O = load_obstacles(n=3)  # Load obstacle configuration
        tm += get_t(A, O, mutant_type)
        tw += get_t(A, O, wild_type)

    # Calculate selection coefficient from survival time
    s = (tm - tw)/tw
    pfix = 2*s  # probability of fixation

    # Draw random n, 0 <= n <= 1
    n = np.random.sample(1)[0]
    if pfix >= n:
        print("Accept mutation")
        return mutant_type
    elif pfix < n:
        print("Reject Mutation")
        return wild_type

def optimise(parameters):
    """Run mutation/selection until optimised function sticks

    To start with, run until 10 mutations are accepted. Check output orbium"""
    mutation_count = 0

    par_in = parameters[:]
    gen_count = 0

    while (mutation_count < 2) & (gen_count < 100):
        par_out = run_one(par_in)
        gen_count += 1
        if par_out != par_in:
            mutation_count += 1
            print(mutation_count, "mutation")
            par_in = par_out[:]

    return par_out

def save_parameters(parameters, filename, cells):
    ### NEED TO FIGURE OUT A SOLUTION TO B
    dict = {}
    keys = ["R", "T", "m", "s", "b"]
    for i in range(len(parameters)):
        dict[keys[i]] = parameters[i]

    with open("results/parameters/parameters_"+filename+".csv", "w") as f:
        csvwrite = csv.writer(f)
        for k in dict:
                csvwrite.writerow([k, dict[k]])
    with open("results/parameters/cells_"+filename+".csv", "w") as f:
        csvwrite = csv.writer(f)
        for i in cells:
            csvwrite.writerow(i)

    return dict

