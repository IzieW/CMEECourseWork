#!/usr/bin/env python3
"""Script to test profiling"""

__author__ = "Izie Wood (iw121@ic.ac.uk)"

########## Functions ##########
def my_squares(iters):
    """Squares numbers in range iters, returns in list"""
    out = []
    for i in range(iters):
        out.append(i ** 2)
    return out 

def my_join(iters, string):
    """Joins numbers in iters with given string"""
    out = ''
    for i in range(iters):
        out += string.join(", ")
    return out

def run_my_func(x, y):
    """Runs above two function"""
    print(x,y)
    my_squares(x)
    my_join(x,y)
    return 0 

run_my_func(10000000, "My string")