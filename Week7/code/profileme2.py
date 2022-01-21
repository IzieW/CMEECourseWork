#!/usr/bin/env python3
"""Script improves on profileme.py- optimising for better run time.
Run with argument -p for profiling"""
__author__ = "Izie Wood (iw121@ic.ac.uk)"

############# imports ##############
import numpy as np


############ Functions #############
def my_squares(iters):
    """Squares each number in iters using numpy array, returns list"""
    out = np.arange(1, iters)  #  array manipulation to optimise run time
    out = out ** 2
    return out

def my_join(iters, string):
    """Joins each number in iters with given string"""
    out = ''
    for i in range(iters):
        out += ", " + string  # Removed 'join' function from profileme1.py to help run time
    return out

def run_my_func(x, y):
    """Runs the above two functions in script"""
    print(x,y)
    my_squares(x)
    my_join(x,y)
    return 0

run_my_func(10000000, "My string")
