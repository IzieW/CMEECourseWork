# !/usr/bin/env python3

"""Script to organise Lenia configurations into cleaner classes.
First work towards developing a package"""

# IMPORTS #
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import csv
from copy import deepcopy
from matplotlib import animation
from scipy.signal import convolve2d
# Silence warnings
np.warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)  # Silence

# CLASSES #
class Creature:
    """Load Lenia life form.
    R: radius
    T: Time
    m: mean (growth function)
    s: standard deviation (growth function)
    b: kernel peaks (growth function)"""
    keys = ["R", "T", "m", "s", "b"]
    size = 64
    mid = size // 2
    cx, cy = 20, 20

    bell = lambda x, m, s:  np.exp(-((x - m) / s) ** 2 / 2)

    def __init__(self, filename):
        """Initiate creature from parameters filename"""
        dict = {}
        # Load parameters #
        with open("parameters/"+filename.lower()+"_parameters.csv", "r") as f:
            csvread = csv.reader(f)
            for row in csvread:
                if row[0] == "b":  # Where b is list of values
                    dict[row[0]] = [float(i) for i in row[1].strip("[]").split(",")]
                else:
                    dict[row[0]] = float(row[1])
        # Load cells #
        cells = []
        with open("parameters/"+filename.lower()+"_cells.csv", "r") as f:
            csvread = csv.reader(f)
            for row in csvread:
                cells.append([float(s) for s in row])
            dict["cells"] = cells

        self.name = filename
        self.config = dict  # Full creature dictionary
        self.theta = [dict[i] for i in ["R", "T", "m", "s", "b"]]  # Parameter vector
        self.cells = dict["cells"]  # Cells only
        # Each parameter
        self.R = self.theta[0]
        self.T = self.theta[1]
        self.m = self.theta[2]
        self.s = self.theta[3]
        self.b = self.theta[4]

    def figure_world(A, cmap="viridis"):
        """Set up basic graphics of unpopulated, unsized world"""
        global img  # make final image global
        fig = plt.figure()  # Initiate figure
        img = plt.imshow(A, cmap=cmap, interpolation="nearest", vmin=0)  # Set image
        plt.title = ("World A")
        plt.close()
        return fig


    def figure_asset(self, cmap="viridis", K_sum=1, bar_K=False):
        """ Chart creature's kernel and growth based on parameter solutions
        Subplot 1: Graph of Kernel in matrix form
        Subplot 2: Cross section of Kernel around center. Y: gives values of cell in row, X: gives column number
        Subplot 3: Growth function according to values of U (Y: growth value, X: values in U)
        """
        R = self.R
        K = self.kernel(fourier=False)
        growth = self.growth
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

    def save(self, verbose=False):
        """Save creature configuration to csv"""
        with open("parameters/" + self.name.lower() + "_parameters.csv", "w") as f:
            csvwrite = csv.writer(f)
            for i in range(len(Creature.keys)):
                csvwrite.writerow([Creature.keys[i], self.theta[i]])
        with open("parameters/"+self.name.lower()+"_cells.csv", "w") as f:
            csvwrite = csv.writer(f)
            for i in self.cells:
                csvwrite.writerow(i)
        if verbose:
        print(self.name+" configuration saved to parameters/")

    def initiate_channel(self, size=size, cx=cx, cy=cy, show=False):
        """Initiate learning channel with creature cell configurations"""
        C = np.asarray(self.cells)  # take cells from creature
        A = np.zeros([size, size])  # Initiate grid of dimensions size x size
        A[cx:cx+C.shape[0], cy:cy+C.shape[1]] = C  # load cell configuration onto grid
        if show:
            plt.matshow(A)
        return A

    def kernel(self, mid=mid, fourier=True, show=False):
        """ Learning kernel for parameter solution. Default fourier transformed"""
        D = np.linalg.norm(np.ogrid[-mid:mid, -mid:mid])/self.R  # define distance matrix
        """Take all cells within distance 1 and transform along gaussian gradient. 
        Produces a smooth ring-shaped kernel"""
        K = (D<1)*Creature.bell(D, 0.5, 0.15)
        K = K/np.sum(K)  # normalise
        if show:
            plt.matshow(K)
        if fourier:
            K = np.fft.fft2(np.fft.fftshift(K))  # fourier transform kernel
        return K

    def growth(self, U):
        """Defines growth of cells based input neighbourhood sum, U
        and parameter configuration. Specifically, mean and standard deviation"""
        return Creature.bell(U, self.m, self.s) * 2 - 1

    def update(self, i):
        """Run stepwise lenia simulation and updates"""
        global A, img
        U = np.real(np.fft.ifft2(K*np.fft.fft2(A)))  # Get neighbourhood sum
        A = np.clip(A + 1/self.T*(self.growth(U)), 0, 1)
        img.set_array(A)
        return img


    def render(self, update=0):
        """Render Creature in environment"""
        if not update:
            update= self.update
        global A, K
        A = self.initiate_channel()
        K = self.kernel()

        fig= Creature.figure_world(A)
        print("Rendering animation...")
        anim = animation.FuncAnimation(fig, update, frames=200, interval=20)
        anim.save("results/"+self.name+"_anim.gif", writer="imagemagick")
        print("Process complete.")

