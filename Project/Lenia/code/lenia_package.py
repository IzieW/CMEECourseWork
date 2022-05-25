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
from IPython.display import HTML, Image
import IPython

# Silence warnings
np.warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)  # Silence


############ INITIATE CLASSES  ####################
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

    bell = lambda x, m, s: np.exp(-((x - m) / s) ** 2 / 2)

    species_cells = {"orbium": [[0, 0, 0, 0, 0, 0, 0.1, 0.14, 0.1, 0, 0, 0.03, 0.03, 0, 0, 0.3, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0.08, 0.24, 0.3, 0.3, 0.18, 0.14, 0.15, 0.16, 0.15, 0.09, 0.2, 0, 0, 0,
                                 0],
                                [0, 0, 0, 0, 0, 0.15, 0.34, 0.44, 0.46, 0.38, 0.18, 0.14, 0.11, 0.13, 0.19, 0.18, 0.45,
                                 0, 0, 0],
                                [0, 0, 0, 0, 0.06, 0.13, 0.39, 0.5, 0.5, 0.37, 0.06, 0, 0, 0, 0.02, 0.16, 0.68, 0, 0,
                                 0],
                                [0, 0, 0, 0.11, 0.17, 0.17, 0.33, 0.4, 0.38, 0.28, 0.14, 0, 0, 0, 0, 0, 0.18, 0.42, 0,
                                 0],
                                [0, 0, 0.09, 0.18, 0.13, 0.06, 0.08, 0.26, 0.32, 0.32, 0.27, 0, 0, 0, 0, 0, 0, 0.82, 0,
                                 0],
                                [0.27, 0, 0.16, 0.12, 0, 0, 0, 0.25, 0.38, 0.44, 0.45, 0.34, 0, 0, 0, 0, 0, 0.22, 0.17,
                                 0],
                                [0, 0.07, 0.2, 0.02, 0, 0, 0, 0.31, 0.48, 0.57, 0.6, 0.57, 0, 0, 0, 0, 0, 0, 0.49, 0],
                                [0, 0.59, 0.19, 0, 0, 0, 0, 0.2, 0.57, 0.69, 0.76, 0.76, 0.49, 0, 0, 0, 0, 0, 0.36, 0],
                                [0, 0.58, 0.19, 0, 0, 0, 0, 0, 0.67, 0.83, 0.9, 0.92, 0.87, 0.12, 0, 0, 0, 0, 0.22,
                                 0.07],
                                [0, 0, 0.46, 0, 0, 0, 0, 0, 0.7, 0.93, 1, 1, 1, 0.61, 0, 0, 0, 0, 0.18, 0.11],
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
                                 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0.02, 0.06, 0.08, 0.09, 0.07, 0.05, 0.01, 0, 0, 0, 0, 0]]}

    def __init__(self, filename, dict=0, species="orbium", cluster=False):
        """Initiate creature from parameters filename, or if file is false, load dictionary"""
        if filename:
            dict = {}
            name = deepcopy(filename)
            # Load parameters #
            if cluster:
                filename = filename.lower()+"_parameters.csv"
            else:
                filename = "../parameters/" + filename.lower() + "_parameters.csv"
            with open(filename, "r") as f:
                csvread = csv.reader(f)
                for row in csvread:
                    if row[0] == "b":  # Where b is list of values
                        dict[row[0]] = [float(i) for i in row[1].strip("[]").split(",")]
                    else:
                        dict[row[0]] = float(row[1])
            self.name = name
        else:
            self.name = dict["orbium"]
        self.cells = Creature.species_cells[species]  # Cells only
        # Each parameter
        self.R = dict["R"]
        self.T = dict["T"]
        self.m = dict["m"]
        self.s = dict["s"]
        self.b = dict["b"]

        self.mutations = 0
        self.evolved_in = 0

        self.survival_mean = dict["survival_mean"]  # Mean survival time in evolved environment
        self.survival_var = dict["survival_var"]  # Survival var in evolved environment

        self.A = 0
        self.K = self.kernel()
        self.enviro = 0  # Load and temporarily hold obstacle channels
        self.enviro_kernel = 0  # Load and temporarily hold obstacle kernels

        self.initiate()  # load A

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

    def save(self, verbose=True, cluster=False):
        """Save creature configuration to csv"""
        if cluster:
            filename = self.name.lower()+"_parameters.csv"
        else:
            filename = "../parameters/" + self.name.lower() + "_parameters.csv"
        with open(filename, "w") as f:
            csvwrite = csv.writer(f)
            for i in Creature.keys:
                csvwrite.writerow([i, self.__dict__[i]])
            csvwrite.writerow(["mutations", self.mutations])
            csvwrite.writerow(["gradient", self.evolved_in])
            csvwrite.writerow(["survival_mean", self.survival_mean])
            csvwrite.writerow(["survival_var", self.survival_var])
        if verbose:
            print(self.name + " configuration saved to parameters/")

    def initiate(self, size=size, cx=cx, cy=cy, show=False):
        """Initiate learning channel with creature cell configurations"""
        C = np.asarray(self.cells)  # take cells from creature
        A = np.zeros([size, size])  # Initiate grid of dimensions size x size
        A[cx:cx + C.shape[0], cy:cy + C.shape[1]] = C  # load cell configuration onto grid
        if show:
            plt.matshow(A)
        self.A = A

    def kernel(self, mid=mid, fourier=True, show=False):
        """ Learning kernel for parameter solution. Default fourier transformed"""
        D = np.linalg.norm(np.ogrid[-mid:mid, -mid:mid]) / self.R  # define distance matrix
        """Take all cells within distance 1 and transform along gaussian gradient. 
        Produces a smooth ring-shaped kernel"""
        K = (D < 1) * Creature.bell(D, 0.5, 0.15)
        K = K / np.sum(K)  # normalise
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
        U = np.real(np.fft.ifft2(self.K * np.fft.fft2(self.A)))  # Convolve by kernel to get neighbourhood sums
        self.A = np.clip(self.A + 1 / self.T * (self.growth(U)), 0, 1)  # Update A by growth function *1/T
        img.set_array(self.A)
        return img,

    def update_obstacle(self, i):
        """Update learning channel by 1/T according to values in the learning channel and obstacle channel"""
        global img
        U = np.real(np.fft.ifft2(self.K * np.fft.fft2(self.A)))  # Convolve by kernel to get neighbourhood sums
        self.A = np.clip(self.A + 1 / self.T * (self.growth(U) + self.obstacle_growth()), 0,
                         1)  # Update A by growth function *1/T
        img.set_array(sum([self.A, self.enviro]))
        return img,

    def update_obstacle_moving(self, i):
        """Update learning channel by 1/T according to values in the learning channel and obstacle channel"""
        global img
        U = np.real(np.fft.ifft2(self.K * np.fft.fft2(self.A)))  # Convolve by kernel to get neighbourhood sums
        self.enviro = convolve2d(self.enviro, self.enviro_kernel, mode="same", boundary="wrap")
        self.A = np.clip(self.A + 1 / self.T * (self.growth(U) + self.obstacle_growth()), 0,
                         1)  # Update A by growth function *1/T
        img.set_array(sum([self.A, self.enviro]))
        return img,

    def render(self, O=0, direction=0, html=False):
        """Render Lenia simulation using above update functions.
        O = obstacle configuration. If specified, renders simulation with this obstacle environment.
        moving = obstacle kernel with desired direction. If specified, renders simulations with moving obstacle environment"""
        self.initiate()
        if type(O) != int:  # Weird logic condition to avoid using "if O.any" since will need to check multiple values in O
            self.enviro = O
            fig = Creature.figure_world(sum([self.A, self.enviro]))
            if type(direction) != int:
                self.enviro_kernel = direction
                print("Rendering animation...")
                if html:
                    IPython.display.HTML(
                        animation.FuncAnimation(fig, self.update_obstacle_moving, frames=200, interval=20).to_jshtml())
                else:
                    anim = animation.FuncAnimation(fig, self.update_obstacle_moving, frames=200, interval=20)
                    anim.save("../results/" + self.name + "_anim.gif", writer="imagemagick")
                print("Process complete.")
            else:
                print("Rendering animation...")
                if html:
                    IPython.display.HTML(
                        animation.FuncAnimation(fig, self.update_obstacle, frames=200, interval=20).to_jshtml())
                else:
                    anim = animation.FuncAnimation(fig, self.update_obstacle, frames=200, interval=20)
                    anim.save("../results/" + self.name + "_anim.gif", writer="imagemagick")
                print("Process complete.")
        else:
            fig = Creature.figure_world(self.A)
            print("Rendering animation...")
            if html:
                IPython.display.HTML(
                    animation.FuncAnimation(fig, self.update_naive, frames=200, interval=20).to_jshtml())
            else:
                anim = animation.FuncAnimation(fig, self.update_naive, frames=200, interval=20)
                anim.save("../results/" + self.name + "_anim.gif", writer="imagemagick")
        # print("Process complete.")

    def render_html(self, o=0):
        if type(o) != int:
            self.enviro = o
            fig = Creature.figure_world(sum([self.A, self.enviro]))
            IPython.display.HTML(
                animation.FuncAnimation(fig, self.update_obstacle, frames=200, interval=20).to_jshtml())
        else:
            fig = Creature.figure_World(self.A)
            IPython.display.HTML(animation.FuncAnimation(fig, self.update_naive, frames=200, interval=20).to_jshtml())

    def update_theta(self, muse):
        """Update parameters from parameters of input instance, muse.
        Update kernel"""
        self.R = muse.R
        self.T = muse.T
        self.s = muse.s
        self.m = muse.m
        self.b = muse.b
        self.K = muse.K

    def theta(self):
        return [self.R, self.T, self.s, self.m, self.b]

    def show(self):
        plt.matshow(self.A)


class ObstacleChannel:
    def __init__(self, n=3, r=5, seed=0, dir="up", gradient=0, peak=1, wrap=True):
        """Defines obstacle environment.
        n = number of obstacles per QUARTER of grid
        r = obstacle radius
        (if moving): dir = direction of movement [up, down, left, right]
        """
        self.name = "enviro_" + str(n) + "_obstacle_radius_" + str(r)
        self.n = n
        self.r = r
        self.seed = seed
        self.gradient = gradient
        self.grid = 0
        self.peak = peak  # max value in obstacle grid
        self.wrap = wrap

        self.direction = dir
        directions = {"up": (0, 1), "down": (2, 1), "left": (1, 0), "right": (1, 2)}
        k = np.zeros([3, 3])
        k[directions[self.direction]] = 1
        self.dir_kernel = k

        mid = Creature.mid
        D = np.linalg.norm(np.ogrid[-mid:mid, -mid:mid]) / self.r

        if gradient:
            exponential = lambda x, l: l * np.exp(-l * x)
            self.kernel = ((D < 1) * exponential(D, self.gradient)) / self.gradient  # normalise to keep between 0 and 1
        else:
            self.kernel = np.ones([self.r, self.r])

        self.initiate()  # Load grid of obstacles

    def figure_asset(self):
        x = np.linspace(0, np.sum(self.kernel), 1000)
        mid = self.kernel.shape[0] // 2
        self.grid = x
        fig, ax = plt.subplots(1, 3)
        ax[0].imshow(self.kernel)
        ax[0].title.set_text("Obstacle kernel")
        ax[1].plot(self.kernel[mid, :])
        ax[1].title.set_text("Kernel cross-section")
        ax[2].plot(x, self.growth())
        ax[2].title.set_text("Obstacle growth")
        plt.show()

    def initiate(self, seed=0, size=Creature.size, mid=Creature.mid):
        """Initiate obstacle channel at random.
        Done by initiating half of grid with random obstacle configurations and
        stiching two halves together to allow more even spacing"""
        if seed:
            np.random.seed(seed)
        o = np.zeros(size * size)
        o[np.random.randint(0, len(o), self.n)] = 1
        o.shape = [size, size]
        # Convolve by kernel shape
        if self.wrap:
            self.grid = convolve2d(o, self.kernel, mode="same", boundary="wrap") * self.peak
        else:
            self.grid = convolve2d(o, self.kernel, mode="same") * self.peak


    def initiate_equal(self, seed=0):
        if seed:
            np.random.seed(seed)
        size = Creature.size // 2
        mid = size // 2
        n = self.n // 4
        o = np.zeros(size * size)
        o[np.random.randint(0, len(o), n)] = 1
        o.shape = [size, size]
        D = np.linalg.norm(np.ogrid[-mid:mid, -mid:mid]) / self.r
        k = (D < 1)
        o = convolve2d(o, k, mode="same", boundary="wrap")  # One half loaded
        o = np.vstack((o, o))
        return np.hstack((o, o))

    def growth(self):
        return -10 * np.maximum(0, (self.grid - 0.001))

    def move(self):
        self.grid = convolve2d(self.grid, self.dir_kernel, mode="same", boundary="wrap")

    def change_dir(self, direction):
        self.direction = direction
        directions = {"up": (0, 1), "down": (2, 1), "left": (1, 0), "right": (1, 2)}
        k = np.zeros([3, 3])
        k[directions[self.direction]] = 1
        self.dir_kernel = k

    def show(self):
        """Show obstacle configuration"""
        plt.matshow(self.grid)


############## INITIATE MEASURES ####################
time_log = pd.DataFrame(columns=["wild_mean", "wild_var", "mutant_mean", "mutant_var"])


def record_time(wild_mean, wild_var, mutant_mean, mutant_var):
    """Record timeline"""
    global time_log
    x = pd.DataFrame([[wild_mean, wild_var, mutant_mean, mutant_var]],
                     columns=["wild_mean", "wild_var", "mutant_mean", "mutant_var"])  # record averages
    time_log = pd.concat([time_log, x])


### EVOLUTION ###
def mutate(p):
    """Mutate input parameter p"""
    return np.exp(np.log(p) + np.random.uniform(low=-0.2, high=0.2))


def prob_fixation(wild_time, mutant_time, N):
    """Return probability of fixation given time survived by mutant and wild type,
    and psuedo-population size N"""
    s = (mutant_time - wild_time) / wild_time  # selection coefficient

    """If s is zero, there is no selective difference between types and probability of 
    fixation is equal to 1/populdation size"""
    if s:
        return (1 - np.exp(-2 * s)) / (1 - np.exp(-2 * N * s))
    else:
        return 1 / N


def update_man(creature, obstacle, moving=False, give_sums=False):
    """Update learning channel by 1/T according to values in learning channel A,
    and obstacle channel O"""
    U = np.real(np.fft.ifft2(creature.K * np.fft.fft2(creature.A)))
    creature.A = np.clip(creature.A + 1 / creature.T * (creature.growth(U) + obstacle.growth()), 0, 1)
    if moving:
        obstacle.move()
    if give_sums:
        return np.sum(creature.A)


def selection(t_wild, t_mutant):
    """Return winning solution based on survival times of wild type (t_wild) and
    mutant type (t_mutant)."""
    pfix = prob_fixation(wild_time=t_wild, mutant_time=t_mutant, N=population_size)  # get probability of fixation
    if pfix >= np.random.uniform(0, 1):
        # ACCEPT MUTATION
        return True
    else:
        # REJECT MUTATION
        return False


def run_one(creature, obstacle, show_after=0, moving=False, verbose=True, give_sums=False):
    """Run creature of given parameters in given obstacle configuration until it dies.
    Show after specifies number of timesteps at when it will show what the grid looks like"""
    t = 0  # set timer
    global sums
    sums = np.zeros(10000)
    while np.sum(creature.A) and (
            t < 10000):  # While there are still cells in the learning channel, and timer is below cut off
        t += 1  # update timer by 1
        if verbose & (t % 1000 == 0):
            print(t)  # Show that it is working even after long waits
        if give_sums:
            sums[t - 1] = update_man(creature, obstacle, moving=moving, give_sums=True)
        else:
            update_man(creature, obstacle, moving=moving)  # Run update and show
        # if t == show_after:
        #   plt.matshow(sum([creature.A, obstacle.grid]))
    return t


def mutate_and_select(creature, obstacle, moving=False, runs=100):
    """Mutate one parameter from creature and assess fitness of new solution agaisnt wild type
    in input obstacle environment. Save winning parameters to Creature.

    Method involve running wild type and mutant over ten distinct obstacle environment, and
    summing the survival time of each."""
    wild_type = creature
    mutant = deepcopy(creature)

    ## Choose parameter at random and mutate in mutant_type
    x = np.random.randint(0, 5)
    mutant.__dict__[Creature.keys[x]] = mutate(mutant.__dict__[Creature.keys[x]])
    mutant.K = mutant.kernel()  # update mutant kernel

    # Run mutant and wild over runs number of obstacle configurations
    t_wild = np.zeros(runs)
    t_mutant = np.zeros(runs)
    for i in range(runs):
        obstacle.initiate()  # configure environment at random
        O = deepcopy(obstacle.grid)
        wild_type.initiate()
        mutant.initiate()
        t_wild[i] = run_one(wild_type, obstacle, moving=moving)
        if moving:
            obstacle.grid = O  # Reset obstacle enviro if obstacles were moved
        t_mutant[i] = run_one(mutant, obstacle, moving=moving)

    # Record mean and variance of survival times
    wild_mean = t_wild.mean()
    print(wild_mean)
    mutant_mean = t_mutant.mean()
    record_time(wild_mean=wild_mean,
                wild_var=t_wild.var(),
                mutant_mean=mutant_mean,
                mutant_var=t_mutant.var())

    # Select winning parameter
    if selection(wild_mean, mutant_mean):
        print("Accept mutation")
        creature.update_theta(mutant)  # Update creature parameters
        return True
    else:
        print("Reject mutation")
        return False


def optimise(creature, obstacle, N, seed=0, fixation=10, moving=False, gradient=False):
    """Mutate and select input creature in psuedo-population of size N
    until wild type becomes fixed over fixation number of generations"""
    global population_size
    population_size = N
    np.random.seed(seed)  # set seed

    """Evolve until parameters become fixed over fixation number of generations"""
    fix = 0  # Initiate fixation count
    while fix < fixation:
        if mutate_and_select(creature, obstacle, moving=moving):  # Updates creature values
            fix = 0  # Mutation has been accepted, reset count
        else:
            fix += 1
            print(fix)

    print("Saving configuration...")
    """Save winning parameters and timelogs"""
    if moving:
        enviro = "moving"
    else:
        enviro = "static"
    if gradient:
        enviro = enviro + "_gradient"
    else:
        enviro = enviro + "_solid"

    creature.name = "orbium" + "_f" + str(fixation) + "_s" + str(seed) + "N" + str(N) + "_enviro_" + enviro

    """Update survival time mean and variance by running over 10 configurations with seed"""
    print("Calculating survival means...")
    survival_time = get_survival_time(creature, obstacle)
    creature.survival_mean = survival_time[0]
    creature.survival_var = survival_time[1]
    creature.evolved_in = obstacle.__dict__

    time_log.to_csv("../results/" + creature.name + "_times.csv")  # Save timelog to csv
    creature.save()  # save parameters
    return 1


import time


def optimise_timely(creature, obstacle, N, seed=0, run_time=10, moving=False, cluster = False):
    """Mutate and select input creature in psuedo-population of size N
    until wild type becomes fixed over fixation number of generations"""
    global population_size
    population_size = N
    np.random.seed(seed)  # set seed

    run_time = run_time * 60  # Translate to seconds
    """Evolve until parameters become fixed over fixation number of generations"""
    gen = 0  # time_count
    mutation = 0
    start = time.time()
    while (time.time() - start) < run_time:
        if mutate_and_select(creature, obstacle, moving=moving):  # Updates creature values
            mutation += 1
        gen += 1

    print("Saving configuration...")
    """Save winning parameters and timelogs"""
    if moving:
        enviro = "m"
    else:
        enviro = "s"
    if obstacle.gradient:
        enviro = enviro + "g" + str(obstacle.gradient) + "r" + str(obstacle.r)
    else:
        enviro = enviro + "s"

    creature.name = "orbium_t" + str(run_time) + "s" + str(seed) + "N" + str(N) + "_enviro_" + enviro

    """Update survival time mean and variance by running over 10 configurations with seed"""
    print("Calculating survival means...")
    survival_time = get_survival_time(creature, obstacle)
    creature.mutations = mutation
    creature.survival_mean = survival_time[0]
    creature.survival_var = survival_time[1]
    creature.evolved_in = obstacle.gradient

    if cluster:
        time_log.to_csv(creature.name + "_times.csv")  # Save timelog to csv
    else:
        time_log.to_csv("../results/" + creature.name + "_times.csv")  # Save timelog to csv
    creature.save(cluster= cluster)  # save parameters
    return 1


def get_survival_time(creature, obstacle, runs=10, summary=False, verbose=False):
    """Calculate average run time over seeded 10 configurations.
    Return mean and variance."""
    times = np.zeros(runs)
    for i in range(1, runs+1):
        creature.initiate()  # Reset grid
        obstacle.initiate(seed=i)  # set obstacle
        times[i - 1] = run_one(creature, obstacle, verbose=verbose)

    if summary:
        return times.mean(), times.var()
    else:
        return times
