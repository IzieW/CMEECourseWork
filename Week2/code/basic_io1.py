#!/usr/bin/env Python3
"""Imports and prints lines from ../data/test.txt"""
__author__ = 'Izie Wood (iw121@ic.ac.uk)'
__version__ = '0.0.1'

#############################
# FILE INPUT
#############################
# Open a file for reading
f = open('../data/test.txt', 'r')  # Sandbox not included in git upload, so moved test file to data directory
# 'r' for 'read file'
# use "implicit" for loop:
# if the object is a file, python will cycle over lines
for line in f: # implicit loop
    print(line)

# close the file
f.close()

# Same example, skip blank lines
f = open('../data/test.txt', 'r')
for line in f:
    if len(line.strip()) > 0:  # Print only if lines are not empty
        print(line)
        
f.close()  # important to always close files to avoid memory leaks
