#!/usr/bin/env python3

"""Script illustrating use of for loops"""
__author__ = 'Izie Wood (iw121@ic.ac.uk)'
__version__ = '0.0.1'

# FOR loops 
for i in range(5):
    print(i)  # prints sequence 1-5

my_list = [0, 2, "geronimo!", 3.0, True, False]
for k in my_list:  # prints each sequence in list
    print (k)

total = 0
summands = [0, 1, 11, 111, 1111]
for s in summands:
    total = total + s
    print(total)

# WHILE loops in python
z = 0
while z < 100:
    z = z + 1
    print(z)

b = 0
while b < 100:
    print("GERONIMO! finite loop! ctrl+c to stop if you must, but it will stop on its own.")
    b = b + 1
# infinite loop made finite for sake of my computer and yours
