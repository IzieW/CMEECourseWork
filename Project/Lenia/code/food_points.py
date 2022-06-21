#!/usr/bin/env python3

"""Develop food source via selection by point collection system.
Food is points. Select for seeking behaviours"""

## IMPORTS ##
from lenia_package import *

class Food:
    def __init__(self, n=1, r=10):
        self.n = n  # number of food blocks
        self.r = r  # Size of food blocks
        self.kernel = np.ones([r, r])
        self.grid = 0

        self.initiate()

    def initiate(self, size=Creature.size, seed=0):
        if seed:
            np.random.seed(seed)
        grid = np.zeros(size * size)
        grid[np.random.randint(0, len(grid), self.n)] = 1
        grid.shape = [size, size]
        # convolve by kernel shape
        self.grid = convolve2d(grid, self.kernel, mode="same", boundary="wrap")

    def show(self):
        plt.imshow(self.grid)
        plt.show()



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


def update_man(creature, food, moving=False, give_sums=False):
    """Update learning channel by 1/T according to values in learning channel A,
    and obstacle channel O"""
    U = np.real(np.fft.ifft2(creature.K * np.fft.fft2(creature.A)))
    creature.A = np.clip(creature.A + 1 / creature.T * (creature.growth(U)), 0, 1)
    if give_sums:
        print(np.sum(creature.A))
