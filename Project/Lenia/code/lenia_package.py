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
    """Defines a life form, their kernel and growth functions
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

    species_cells = {"orbium":[[0, 0, 0, 0, 0, 0, 0.1, 0.14, 0.1, 0, 0, 0.03, 0.03, 0, 0, 0.3, 0, 0, 0, 0],
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
                    [0, 0, 0, 0, 0, 0, 0, 0, 0.02, 0.06, 0.08, 0.09, 0.07, 0.05, 0.01, 0, 0, 0, 0, 0]]}

    def __init__(self, filename, dict=0, species="orbium"):
        """Initiate creature from parameters filename, or if file is false, load dictionary"""
        if filename:
            dict = {}
            # Load parameters #
            with open("parameters/"+filename.lower()+"_parameters.csv", "r") as f:
                csvread = csv.reader(f)
                for row in csvread:
                    if row[0] == "b":  # Where b is list of values
                        dict[row[0]] = [float(i) for i in row[1].strip("[]").split(",")]
                    else:
                        dict[row[0]] = float(row[1])
            self.name = filename
        else:
            self.name=dict["name"]
        self.cells = Creature.species_cells[species]  # Cells only
        # Each parameter
        self.R = dict["R"]
        self.T = dict["T"]
        self.m = dict["m"]
        self.s = dict["s"]
        self.b = dict["b"]

        self.A = self.initiate()
        self.K = self.kernel()
        self.enviro = 0  # Load and temporarily hold obstacle channels
        self.enviro_kernel = 0  # Load and temporarily hold obstacle kernels

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

    def initiate(self, size=size, cx=cx, cy=cy, show=False):
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

    def obstacle_growth(self):
        """Obstacle growth function: Obstacle creates severe negative growth in Life form"""
        return -10 * np.maximum(0, (self.enviro - 0.001))


    def update_naive(self, i):
        """Update learning channel by 1/T according to values in the learning channel"""
        global img
        U = np.real(np.fft.ifft2(self.K*np.fft.fft2(self.A)))  # Convolve by kernel to get neighbourhood sums
        self.A = np.clip(self.A + 1/self.T*(self.growth(U)), 0, 1)  # Update A by growth function *1/T
        img.set_array(self.A)
        return img,

    def update_obstacle(self, i):
        """Update learning channel by 1/T according to values in the learning channel and obstacle channel"""
        global img
        U = np.real(np.fft.ifft2(self.K*np.fft.fft2(self.A)))  # Convolve by kernel to get neighbourhood sums
        self.A = np.clip(self.A + 1/self.T*(self.growth(U)), 0, 1) # + self.obstacle_growth()), 0, 1)  # Update A by growth function *1/T
        img.set_array(sum([self.A, self.enviro]))
        return img,


    def update_obstacle_moving(self, i):
        """Update learning channel by 1/T according to values in the learning channel and obstacle channel"""
        global img
        U = np.real(np.fft.ifft2(self.K*np.fft.fft2(self.A)))  # Convolve by kernel to get neighbourhood sums
        self.enviro = convolve2d(self.enviro, self.enviro_kernel, mode="same", boundary="wrap")
        self.A = np.clip(self.A + 1/self.T*(self.growth(U) + self.obstacle_growth()), 0, 1)  # Update A by growth function *1/T
        img.set_array(sum([self.A, self.enviro]))
        return img,

    def render(self, O=0, direction=0):
        """Render Lenia simulation using above update functions.
        O = obstacle configuration. If specified, renders simulation with this obstacle environment.
        moving = obstacle kernel with desired direction. If specified, renders simulations with moving obstacle environment"""
        if type(O) != int:  # Weird logic condition to avoid using "if O.any" since will need to check multiple values in O
            self.enviro = O
            fig = Creature.figure_world(sum([self.A, self.enviro]))
            if type(direction) != int:
                self.enviro_kernel = direction
                print("Rendering animation...")
                anim = animation.FuncAnimation(fig, self.update_obstacle_moving, frames=10, interval=20)
                anim.save("results/"+self.name+"_anim.gif", writer="imagemagick")
                print("Process complete.")
            else:
                print("Rendering animation...")
                anim = animation.FuncAnimation(fig, self.update_obstacle, frames=10, interval=20)
                anim.save("results/"+self.name+"_anim.gif", writer="imagemagick")
                print("Process complete.")
        else:
            fig= Creature.figure_world(self.A)
            print("Rendering animation...")
            anim = animation.FuncAnimation(fig, self.update_naive, frames=10, interval=20)
            anim.save("results/"+self.name+"_anim.gif", writer="imagemagick")
            print("Process complete.")


class ObstacleChannel:
    def __init__(self, n = 3, r= 5, seed=0, dir="up", gradient = 1):
        """Defines obstacle environment.
        n = number of obstacles per QUARTER of grid
        r = obstacle radius
        (if moving): dir = direction of movement [up, down, left, right]
        """
        self.name = "enviro_"+str(n)+"_obstacle_radius_"+str(r)+"_seed_"+str(seed)
        self.n = n
        self.r = r
        self.seed = seed
        self.gradient = gradient

        k = np.zeros([3,3])
        directions = {"up": (0,1), "down":(2, 1), "left":(1, 0), "right":(1, 2)}

        if dir:
            k[directions[dir]] = 1
            self.kernel = k


    def initiate(self, gradient=False, seed = False, size= Creature.size, mid = Creature.mid):
        """Initiate obstacle channel at random.
        Done by initiating half of grid with random obstacle configurations and
        stiching two halves together to allow more even spacing"""
        if seed:
            np.random.seed(self.seed)
        o = np.zeros(size*size)
        o[np.random.randint(0, len(o), self.n)] = 1
        o.shape = [size, size]
        # Convolve by one of two kernels to get desired shape with radius self.r
        D = np.linalg.norm(np.ogrid[-mid:mid, -mid:mid])/self.r
        if gradient:
            D = D/2
            exponential = lambda x, l: l*np.exp(-l*x)
            k = (D<1)* exponential(D, self.gradient)
        else:
            k = (D<1)
        return convolve2d(o, k, mode="same", boundary="wrap")









