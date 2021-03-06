#!/usr/bin/env Python3

"""Illustrates exporting data from lists to files.
Saves list of 1-100 to file."""

__author__ = 'Izie Wood (iw121@ic.ac.uk)'
__version__ = '0.0.1'

#############################
# FILE OUTPUT
#############################
# save the elements of a list to a file
list_to_save = range(100)

f = open('../results/testout.txt', 'w')  # 'w' for 'write'
for i in list_to_save:
    f.write(str(i) + '\n')  # Add a new line at the end

f.close()  # close file

print("Done! List saved to ../results/testout.txt")  # Feedback to user
