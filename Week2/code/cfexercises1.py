#!/usr/bin/env python3

"""New and improved cfexercises1.py modified to make it a module. 
All the foo_x functions will now take arguments from the user"""

__author__ = 'Izie Wood (iw121@ic.ac.uk)'
__version__ = '0.0.1'

## imports ##
import sys 

## functions ##
def foo_1(x): # Returns x to the power of 0.05
    """Returns number to the power of 0.05"""
    return x **0.5 

def foo_2(x, y): # Returns the greater variable
    """Returns the greater value"""
    if x > y:
        return x
    return y

def foo_3(x, y, z): # Reorders x and y according to which is bigger, then reorders y and x according to which is bigger.
    """Reorders according to greatest value"""
    if x > y:
        tmp = y
        y = x
        x = tmp
    if y > z:
        tmp = z
        z = y
        y = tmp
    return [x, y, z]

def foo_4(x): # calculates the factorial of x
    """Calculates the factorial of x"""
    result = 1
    for i in range(1, x + 1):
        result = result * i
    return result

def foo_5(x): # a recursive function that calculates the factorial of x 
    """A recursive function that Calculates the factorial of x"""
    if x == 1:
        return 1
    return x * foo_5(x - 1)

def foo_6(x): # Calculate the factorial of x in a different way
    """Calculates the factorial of x in a different way"""
    facto = 1
    while x >= 1:
        facto = facto * x
        x = x -1
    return facto

def main(argv):
    """Give values and print outputs of functions above"""
    print(foo_1(2))
    print(foo_2(1,2))
    print(foo_3(5,4,3))
    print(foo_4(4))
    print(foo_5(4))
    print(foo_6(4))
    return 0 

if (__name__ == "__main__"):
    status = main(sys.argv)
    sys.exit(status)