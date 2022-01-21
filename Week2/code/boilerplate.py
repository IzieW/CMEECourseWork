#!/usr/bin/env python3
"""Simple boilerplate example program for python"""

__appname__ = '[boilerplate.py]'
__author__ = 'Izie Wood (iw121@ic.ac.uk)'
__version__ = '0.0.1'
__license__ = "License for this code/program"

## imports ##
import sys  # module to interface our programes with the operating system


## constants ##

## functions ##
def main(argv):
    """Main entry point of the program"""
    print('This is a boilerplate')  # NOTE: indented using two tabs or 4 spaces
    return 0


if __name__ == "__main__":
    """Make sure the main function is called from the command line"""
    status = main(sys.argv)
    sys.exit(status)
