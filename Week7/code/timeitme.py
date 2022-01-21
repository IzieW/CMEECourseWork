#!/usr/bin/env python
"""
Script for illustrating use of timeit module in python for
making timewise comparisons between different functionalities.

Script imports functions from past two profileme scripts, to be run in console
using timeit(): 

%timeit FUNCTION(string)

Returns system runtime for each function called. 
"""

__author__ = "Izie Wood (iw121@ic.ac.uk)"

#########################################################
# loops vs. np.array comprehsensions: which is faster?
#########################################################

iters = 1000000

import timeit  # Module allows targetted run-time measurements of specific functions

from profileme import my_squares as my_squares_loops

from profileme2 import my_squares as my_squares_array

###############################################################
# loops vs. the join method for strings: which is faster
################################################################
mystring = "my string"

from profileme import my_join as my_join_join

from profileme2 import my_join as my_join

# Run %timeit my_squares_loops(iters) in console! See which takes longer.
