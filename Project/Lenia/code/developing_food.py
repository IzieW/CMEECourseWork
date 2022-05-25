# !/usr/bin/env python3

"""Script begins exploring introducing a food source to
lenia creatures."""

from lenia_package import *

### ATTEMPT 1: Naive food  ###
"""Food blocks modelled like obstacles. Passage over creates growth in cells."""

class Food:
    def __init__(self, n=1, r=10):
        self.n = n
        self.r = r
        self.kernel = np.ones([r, r])
        self.grid = 0

        self.initiate()

    def initiate(self, size = Creature.size, seed=0):
        if seed:
            np.random.seed(seed)
        grid = np.zeros(size*size)
        grid[np.random.randint(0, len(grid), self.n)]  = 1
        grid.shape = [size, size]
        # convolve by kernel shape
        self.grid = convolve2d(grid, self.kernel, mode="same", boundary="wrap")

    def show(self):
        plt.imshow(self.grid)
        plt.show()

    def growth(self):
        return 0.5*self.grid

