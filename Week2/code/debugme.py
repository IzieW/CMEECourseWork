#!/usr/bin/env python3

"""Buggy function to run pdb on"""
__author__ = 'Izie Wood (iw121@ic.ac.uk)'
__version__ = '0.0.1'


def buggyfunc(x):
    """Function to debug"""
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


buggyfunc(20)
