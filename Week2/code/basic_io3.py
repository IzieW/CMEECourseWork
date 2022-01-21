#!/usr/bin/env python3

"""Stores dictionary to file, then reloads it using pickle"""

__author__ = 'Izie Wood (iw121@ic.ac.uk)'
__version__ = '0.0.1'

#############################
# STORING OBJECTS
#############################
# to save an object (even complex) for later use
my_dictionary = {"a key": 10, "another key": 11}

import pickle

f = open('../results/testp.p', 'wb')  # wb - "write binary"
pickle.dump(my_dictionary, f)  # save to file f
f.close()

print("Dictionary saved to ../results/testp.p")

## Load the data again
print("Loading .../results/testp.p...")
f = open('../results/testp.p', 'rb')  # read binary
another_dictionary = pickle.load(f)  ## Assign data from f to another_dictionary
f.close()

print("Loaded dictionary is: ", another_dictionary)
