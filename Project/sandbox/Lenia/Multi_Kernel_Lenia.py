# !/usr/bin/env python3
"""Execute Lenia with multiple Kernels"""

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

def figure_asset_list(Ks, nKs, growth, kernels, use_c0=False, cmap='viridis', K_sum=1):
  global R
  K_size = Ks[0].shape[0];  K_mid = K_size // 2
  fig, ax = plt.subplots(1, 3, figsize=(14,2), gridspec_kw={'width_ratios': [1,2,2]})
  if use_c0:
    K_stack = [ np.clip(np.zeros(Ks[0].shape) + sum(K/3 for k,K in zip(kernels,Ks) if k['c0']==l), 0, 1) for l in range(3) ]
  else:
    K_stack = Ks[:3]
  ax[0].imshow(np.dstack(K_stack), cmap=cmap, interpolation="nearest", vmin=0)
  ax[0].title.set_text('kernels Ks')
  X_stack = [ K[K_mid,:] for K in nKs ]
  ax[1].plot(range(K_size), np.asarray(X_stack).T)
  ax[1].title.set_text('Ks cross-sections')
  ax[1].set_xlim([K_mid - R - 3, K_mid + R + 3])
  x = np.linspace(0, K_sum, 1000)
  G_stack = [ growth(x, k['m'], k['s']) * k['h'] for k in kernels ]
  ax[2].plot(x, np.asarray(G_stack).T)
  ax[2].axhline(y=0, color='grey', linestyle='dotted')
  ax[2].title.set_text('growths Gs')
  return fig

## Load creatures
pattern = {}
"""Fish parameters includes list of kernels and growth parameters"""
pattern["fish"] = {"name": "K=3 Fish", "R": 10, "T": 5, "kernels": [
    {"b": [1, 5 / 12, 2 / 3], "m": 0.156, "s": 0.0118, "h": 1, "c0": 0, "c1": 0},
    {"b": [1 / 12, 1], "m": 0.193, "s": 0.049, "h": 1, "c0": 0, "c1": 0},
    {"b": [1], "m": 0.342, "s": 0.0891, "h": 1, "c0": 0, "c1": 0}],
                   "cells": [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.06, 0.1, 0.04, 0.02, 0.01, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0.15, 0.37, 0.5, 0.44, 0.19, 0.23, 0.3, 0.23, 0.15, 0.01, 0, 0, 0,
                              0], [0, 0, 0, 0, 0, 0, 0.32, 0.78, 0.26, 0, 0.11, 0.11, 0.1, 0.08, 0.18, 0.16, 0.17, 0.24,
                                   0.09, 0, 0, 0],
                             [0, 0, 0, 0, 0.45, 0.16, 0, 0, 0, 0, 0, 0.15, 0.15, 0.16, 0.15, 0.1, 0.09, 0.21, 0.24,
                              0.12, 0, 0],
                             [0, 0, 0, 0.1, 0, 0, 0, 0, 0, 0, 0, 0.17, 0.39, 0.43, 0.34, 0.25, 0.15, 0.16, 0.15, 0.25,
                              0.03, 0],
                             [0, 0.15, 0.06, 0, 0, 0, 0, 0, 0, 0, 0.24, 0.72, 0.92, 0.85, 0.61, 0.47, 0.39, 0.27, 0.12,
                              0.18, 0.17, 0],
                             [0, 0.08, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 1.0, 1.0, 1.0, 0.73, 0.6, 0.56, 0.31, 0.12, 0.15,
                              0.24, 0.01],
                             [0, 0.16, 0, 0, 0, 0, 0, 0, 0, 0.76, 1.0, 1.0, 1.0, 1.0, 0.76, 0.72, 0.65, 0.39, 0.1, 0.17,
                              0.24, 0.05],
                             [0, 0.05, 0, 0, 0, 0, 0, 0, 0.21, 0.83, 1.0, 1.0, 1.0, 1.0, 0.86, 0.85, 0.76, 0.36, 0.17,
                              0.13, 0.21, 0.07],
                             [0, 0.05, 0, 0, 0.02, 0, 0, 0, 0.4, 0.91, 1.0, 1.0, 1.0, 1.0, 1.0, 0.95, 0.79, 0.36, 0.21,
                              0.09, 0.18, 0.04],
                             [0.06, 0.08, 0, 0.18, 0.21, 0.1, 0.03, 0.38, 0.92, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.64,
                              0.31, 0.12, 0.07, 0.25, 0],
                             [0.05, 0.12, 0.27, 0.4, 0.34, 0.42, 0.93, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.97,
                              0.33, 0.16, 0.05, 0.1, 0.26, 0],
                             [0, 0.25, 0.21, 0.39, 0.99, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.86, 0.89, 0.94, 0.83, 0.13, 0,
                              0, 0.04, 0.21, 0.18, 0],
                             [0, 0.06, 0.29, 0.63, 0.84, 0.97, 1.0, 1.0, 1.0, 0.96, 0.46, 0.33, 0.36, 0, 0, 0, 0, 0,
                              0.03, 0.35, 0, 0],
                             [0, 0, 0.13, 0.22, 0.59, 0.85, 0.99, 1.0, 0.98, 0.25, 0, 0, 0, 0, 0, 0, 0, 0, 0.34, 0.14,
                              0, 0],
                             [0, 0, 0, 0, 0.33, 0.7, 0.95, 0.8, 0.33, 0.11, 0, 0, 0, 0, 0, 0, 0, 0.11, 0.26, 0, 0, 0],
                             [0, 0, 0, 0, 0.16, 0.56, 0.52, 0.51, 0.4, 0.18, 0.01, 0, 0, 0, 0, 0, 0, 0.42, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0.01, 0, 0.33, 0.47, 0.33, 0.05, 0, 0, 0, 0, 0, 0, 0.35, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0.26, 0.32, 0.13, 0, 0, 0, 0, 0, 0, 0, 0.34, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0.22, 0.25, 0.03, 0, 0, 0, 0, 0, 0, 0.46, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0.09, 0.2, 0.22, 0.23, 0.23, 0.22, 0.3, 0.3, 0, 0, 0, 0, 0, 0, 0, 0]]
                   }

###### MULTIPLE KERNELS #####
"""Extend system into multiple Kernels and growth functions. 
Multiply properties (Kernels, neighbourhood sum U, growth values)
into lists, and list comprehension"""

# SET UP
bell = lambda x, m, s: np.exp(-((x - m) / s) ** 2 / 2)  # Gaussian function
size = 64;
mid = size // 2;
scale = 1;
cx, cy = 20, 20
globals().update(pattern["fish"]);
C = np.asarray(cells)  # Load fish

A = np.zeros([size, size])
C = sc.ndimage.zoom(C, scale, order=0);
R *= scale
A[cx:cx + C.shape[0], cy:cy + C.shape[1]] = C  # Load in fish cells to grid

""" Prepare multiple Kernels with list comphrehension"""
# D=
# K=
# fK=
# Get distance matrix for each set of b in fish kernels
Ds = [np.linalg.norm(np.ogrid[-mid:mid, -mid:mid]) / R * len(k['b']) for k in kernels]
Ks = [(D < len(k["b"])) * np.asarray(k["b"])
[np.minimum(D.astype(int), len(k["b"]) - 1)] * bell(D % 1, 0.5, 0.15) for D, k in zip(Ds, kernels)]

nKs = [ K / np.sum(K) for K in Ks]  # Normalise each Kernel in list
fKs = [ np.fft.fft2(np.fft.fftshift(k)) for k in nKs]  # Fourier transform each kernel in list

def growth(U, m, s):
    return bell(U, m, s)*2-1

def update(i):
    global A, img
    """Use multiple kernels, calculate average growth"""
    Us = [ np.real(np.fft.ifft2(fk * np.fft.fft2(A))) for fk in fKs]  # Convolve A by each Kernel
    Gs = [ growth(U, k["m"], k["s"]) for U, k in zip (Us, kernels)]  # Calculate growth for each Kernel with each set of growth parameters (m,s )
    A = np.clip(A + 1/T * np.mean(np.asarray(Gs), axis=0), 0, 1)  # Multiply A by growth means and clip
    img.set_array(A)
    return img,

## Check out individual kernels
b_list = [ k["b"] for k in kernels]
fig, ax = plt.subplots(1, 3)
ax[0].imshow(Ks[0])
ax[1].imshow(Ks[1])
ax[2].imshow(Ks[2])
plt.show()

# All kernels together
figure_asset_list(Ks, nKs, growth, kernels)
plt.show()

fig = figure_world(A)
anim = animation.FuncAnimation(fig, update, frames=200, interval= 20)
anim.save("results/multi_kernel.gif", writer = "imagemagick")
