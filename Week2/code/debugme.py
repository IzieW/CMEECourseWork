#!/usr/bin/env python3

"""
Expanded buggy function that uses try and except key words to 
anticipate errors in program, and allow code to run despite them
"""
__author__ = 'Izie Wood (iw121@ic.ac.uk)'
__version__ = '0.0.1'


def buggyfunc(x):
    """Manipulates variales of input range.
    Catches errors without stopping function"""
    y = x
    for i in range(x):
        try:
            y = y - 1
            z = x / y
        except ZeroDivisionError:
            print(f"The result of dividing a number by zero is undefined")
        except:
            print(f"This didn't work; x= {x}; y = {y}")
        else:
            print(f"OK; x = {x}; y = {y}, z = {z};")
    return z


buggyfunc(20)  # call function
