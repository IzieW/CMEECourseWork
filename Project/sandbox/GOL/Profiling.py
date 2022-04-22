# !/usr/bin/env python3
"""Script for misc profiling and optimisation"""
import numpy as np
import scipy as sc


def life_step_1(x):
    """Game of life step using generator expression"""
    nbrs_count = sum(np.roll(np.roll(x, i, 0), j, 1)
                     for i in (-1, 0, 1) for j in (-1, 0, 1)
                     if (i != 0 or j != 0))
    return (nbrs_count == 3) | (x & (nbrs_count == 2))


def life_step_2(x):
    """Game of life using scipy tools"""
    from scipy.signal import convolve2d
    nbrs_count = convolve2d(x, np.ones((3, 3)), mode="same", boundary='wrap') - x
    return (nbrs_count == 3) | (x & (nbrs_count == 2))


x = np.random.randint(2, size=(60, 60))

life_step_1(x)
life_step_2(x)
