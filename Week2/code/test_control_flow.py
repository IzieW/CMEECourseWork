#!/usr/bin/env python3
"""Exercise using doctest tool to test function from control_flow.py"""
#docstrings are considered part of the running code (normal comments are #stripped). Hence, you can access your docstrings at run time. 

__appname__ = '[test_control_flow.py]'
__author__ = 'Izie Wood (iw121@ic.ac.uk)'
__version__ = '0.0.1'
__liscense__ = "License for this code/program"

## imports ##
import sys # module to interface our programes with the operating system 
import doctest # Import the doctest module

## constants ##

## functions ##
def even_or_odd(x=0):
    """Doctest even or odd: Find whether a number x is even or odd.

    >>> even_or_odd(10)
    '10 is Even!'

    >>> even_or_odd(5)
    '5 is Odd!'

    whenever a float is provided, then the closest integer is used:
    >>> even_or_odd(3.2)
    '3 is Odd!'

    in case of negative numbers, the positive is taken:
    >>> even_or_odd(-2)
    '-2 is Even!'

    """
    # Define the function to be tested
    if x % 2 == 0:
        return "%d is Even!" % x
    return "%d is Odd!" % x

def main(argv):
    print(even_or_odd(22))
    print(even_or_odd(33))
    return 0

if (__name__ == "__main__"):
    status = main(sys.argv)

doctest.testmod() # To run with embedded tests
