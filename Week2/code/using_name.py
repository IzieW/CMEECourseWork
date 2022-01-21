#!/usr/bin/env python3


"""Script illustrating transfer of control when using 
if __name__ == '__main__' in a program"""

__appname__ = '[using_name.py]'
__author__ = 'Izie Wood (iw121@ic.ac.uk)'
__version__ = '0.0.1'

if __name__ == '__main__':  # When program is called from command line..
    print('This program is being run by itself')
else:  # When program is imported as a module
    print('I am being imported from another module')

print("This module's name is "+ __name__)  # See how name changes based on how it is called

# Specification useful when importing programs as modules. Allows us to return/run certain
# functions only when called from command line, to avoid unwanted outputs when importing module.
