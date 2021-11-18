#!/usr/bin/env python3
"""Script to profile"""
__author__ = "Izie Wood (iw121@ic.ac.uk)"

############# imports ##############
import numpy as np


############ Functions #############
def my_squares(iters):
    """Squares each number in iters using numpy array, returns list"""
    out = np.arange(1, iters)
    out = out ** 2
    return out

def my_join(iters, string):
    """Joins each number in iters with given string"""
    out = ''
    for i in range(iters):
        out += ", " + string
    return out

def run_my_func(x, y):
    """Runs the above two functions in script"""
    print(x,y)
    my_squares(x)
    my_join(x,y)
    return 0

run_my_func(10000000, "My string")