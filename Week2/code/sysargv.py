#!/usr/bin/env python3

"""Script illustrating function of sys.argv in python. 
Prints name of script, number of arguments that are passed to script,
and the arguments that were passed"""
__appname__ = '[sys.argv.py]'
__author__ = 'Izie Wood (iw121@ic.ac.uk)'
__version__ = '0.0.1'

import sys

# argv holds the arguments passed to script from command line

print("This is the name of the script: ", sys.argv[0])
print("Number of arguments: ", len(sys.argv))
print("The arguments are: ", str(sys.argv))
