#!/usr/bin/env python3

"""More functions illustrating use of control loops"""
__author__ = 'Izie Wood (iw121@ic.ac.uk)'
__version__ = '0.0.1'


def hello_1(x):
    """Prints hello when numbers are divisible by 3"""
    for j in range(x):
        if j % 3 == 0:  # % remainder
            print('hello')
    print(' ')


hello_1(12)


def hello_2(x):
    """Print hello when numbers divided by 5 have a remainder of 3"""
    for j in range(x):
        if j % 5 == 3:
            print('hello')
        elif j % 4 == 3:
            print('hello')
    print(' ')


hello_2(12)


def hello_3(x, y):
    """Prints hello for as many numbers as are in a given range"""
    for i in range(x, y):
        print('hello')
    print(' ')


hello_3(3, 17)


def hello_4(x):
    """Print hello until x equals 15"""
    while x != 15:
        print('hello')
        x = x + 3
    print(' ')


hello_4(0)


def hello_5(x):
    """ Print hello if x equals 31 or 18"""
    while x < 100:
        if x == 31:
            for k in range(7):
                print('hello')
        elif x == 18:
            print('hello')
        x = x + 1
    print(' ')


hello_5(12)


# WHILE loop with BREAK

def hello_6(x, y):
    """Prints hello and y, until y=6"""
    while x:
        print("hello!" + str(y))
        y += 1  # increment by 1
        if y == 6:
            break
    print(' ')


hello_6(True, 0)
